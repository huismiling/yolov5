
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

import magicmind.python.runtime as mm
from magicmind.python.runtime import Context


def build_model(mm_file_name):
    model = mm.Model()
    model.deserialize_from_file(mm_file_name)
    assert model != None, "Failed to build model"
    return model

# 创建推理模型所需的上下文
def create_context(model, mm_dump):
    econfig = mm.Model.EngineConfig()
    econfig.device_type = "MLU"
    engine = model.create_i_engine(econfig)
    assert engine != None, "Failed to create engine"
    context = engine.create_i_context()
    
    if mm_dump:
        dumpinfo = Context.ContextDumpInfo(path="/tmp/output_pb/", tensor_name=[], dump_mode=1, file_format=0)
        # dumpinfo.val.dump_mode = 1   # -1 关闭dump 模式; 0 dump 指定tensor; 1 dump所有tensor; 2 dump 输出tensor
        # dumpinfo.val.path = "/tmp/output"# 将dump 结果存放到/tmp/output ⽬录下
        # dumpinfo.val.tensor_name = [] # dump 所有tensor 信息
        # dumpinfo.val.file_format = 0 # 0 ⽂件保存为pb; 1 ⽂件保存为pbtxt
        context.set_context_dump_info(dumpinfo)

    return context

class MagicMindModel():
    """
    MagicMind model use .model file.
    """

    def __init__(self, mm_file_name, mm_dump=False, device_id=0):
        dev = mm.Device()
        dev.id = device_id
        assert dev.active().ok(), "device error"
        #创建model
        model = build_model(mm_file_name)
        self.system = mm.System()
        self.system.initialize()   #  auto destory
        # 创建运行模型时的上下文
        context = create_context(model, mm_dump)
        # 创建队列
        queue = dev.create_queue()
        assert queue != None
        # 创建input
        inputs = context.create_inputs()
        assert type(inputs) != mm.Status
        self.dev = dev
        self.context =context
        self.queue = queue
        self.inputs = inputs
        self.batch_size = model.get_input_dimension(0).GetDimValue(0)

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        self.inputs[0].from_numpy(x)
        self.inputs[0].to(self.dev)
        # 创建output
        outputs = []
        assert type(outputs) != mm.Status
        for out in outputs:
            out.to(self.dev)
        # 发送任务
        status = self.context.enqueue(self.inputs, outputs, self.queue)
        # 阻塞队列，直至得到运行结果
        self.queue.sync()
        assert status.ok(), "inference error"

        return torch.from_numpy(outputs[0].asnumpy())

    def __call__(self, x):
        return self.forward(x)

    def eval(self):
        return self


