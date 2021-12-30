import os
import json
import numpy as np
import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime import Context
from calibrator_custom_data import FixedCalibData

parser = argparse.ArgumentParser()

# 基础参数，必选
parser.add_argument("--onnx", help="", type=str, default="yolov5_m.onnx")
parser.add_argument("--mm_file_name", help="", type=str, default="yolov5_m_int8fp16.model")
parser.add_argument("--quant_datasets", help="", type=str, default="quant_datasets/")
# quant_mode supports 
# "qint8_mixed_float32","qint8_mixed_float16", 
# "qint16_mixed_float32", "qint16_mixed_float16", 
# "force_float16" and "force_float32".
parser.add_argument("--quant_mode", help="", type=str, default="qint8_mixed_float16")
parser.add_argument("--input_shapes", help="", type=list, default=[[1, 3, 640, 640]])
parser.add_argument("--input_dtypes", help="", type=list, default=["float32"])

args = parser.parse_args()

# 解析模型
def parse_model():
    network = mm.Network()
    parser = mm.Parser(mm.ModelKind.kOnnx)
    status = parser.parse(network, args.onnx)
    assert status.ok(), "parser network error"
    for i in range(len(args.input_shapes)):
        network.get_input(i).set_dimension(mm.Dims(args.input_shapes[i]))
    return network

# 生成模型
def build_model(network):

    config = mm.BuilderConfig()
    # config.parse_from_file(args.builder_config)
    cfg_dict = {
                    "archs": ["mtp_270"],
                    "graph_shape_mutable": False,
                    "opt_config": {
                        "type64to32_conversion": True,
                        "conv_scale_fold": False
                    },
                    "file_path": "dump_file",
                    "debug_config": {
                        "print_ir": {
                            "before_build": True,
                            "after_build": True
                        }
                    }
                }
    config.parse_from_string(json.dumps(cfg_dict))

    if "*" in args.quant_datasets or \
       os.path.isdir(args.quant_datasets) or \
       os.path.isfile(args.quant_datasets):
        # get input dims from network
        input_dims = network.get_input(0).get_dimension()

        # create calibrate data
        calib_data = FixedCalibData(shape = input_dims,
                                    data_type = mm.DataType.FLOAT32,
                                    max_samples = 10,
                                    data_paths = args.quant_datasets)

        # create calibrator
        calibrator = mm.Calibrator([calib_data])
        assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
        precision_dict = {
                        "precision_config": {
                            "precision_mode": args.quant_mode,
                            "weight_quant_granularity": "per_axis",
                            "activation_quant_algo": "symmetric",
                        }
                    }
        assert config.parse_from_string(json.dumps(precision_dict)).ok()

        # calibrate the network
        calibrator.calibrate(network, config)

    for itn in ['472', '531', '590']:
        itensor = network.find_tensor_by_name(itn)
        network.unmark_output(itensor)
    builder = mm.Builder()
    model = builder.build_model("test", network, config)
    assert model != None, "Failed to build model"
    return model

# 创建推理模型所需的上下文
def create_context(model, dev):
    econfig = mm.Model.EngineConfig()
    econfig.device_type = "MLU"
    engine = model.create_i_engine(econfig)
    assert engine != None, "Failed to create engine"
    context = engine.create_i_context()

    return context

if __name__ == '__main__':
    # 指定要使用的mlu设备
    dev = mm.Device()
    dev.id = 0
    assert dev.active().ok(), "device error"
    # 创建network
    network = parse_model()
    #创建model
    model = build_model(network)
    print(model.get_output_names())
    model.serialize_to_file(args.mm_file_name)
    with mm.System():
        # 创建运行模型时的上下文
        context = create_context(model, dev)
        # 创建队列
        queue = dev.create_queue()
        assert queue != None
        # 创建input
        inputs = context.create_inputs()
        assert type(inputs) != mm.Status
        # 传入需要推理的数据
        for i in range(len(inputs)):
            data = np.random.normal(-1, 1, size=args.input_shapes[i]).astype(args.input_dtypes[i])
            inputs[i].from_numpy(data)
            inputs[i].to(dev)
        # 创建output
        outputs = context.create_outputs(inputs)
        assert type(outputs) != mm.Status
        for out in outputs:
            out.to(dev)
        # 发送任务
        status = context.enqueue(inputs, outputs, queue)
        # 阻塞队列，直至得到运行结果
        queue.sync()
        assert status.ok(), "inference error"
        for tensor in outputs:
            print(tensor.shape)

