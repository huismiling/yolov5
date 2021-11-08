# YOLOv5-Python-MagicMind

## Tutorial

### Step1
Install MagicMind relative software and python wheel. Please following the reference https://developer.cambricon.com.

### Step2
Use provided tools to generate onnx file.
For example, if you want to generate onnx file of yolox-m, please run the following command:
```shell
cd <path of yolov5>
python export.py --weights weights/yolov5m.pt --include onnx
```
Then, a weights/yolov5m.onnx file is generated.

### Step3
Generate MagicMind model.
```shell
cd MagicMind/python/
python onnx2mm.py --onnx  ../../weights/yolov5m.onnx --quant_datasets quant_datasets/
```
Then, a yolox_m_int8fp16.model file is generated.

### Step4
Use MagicMind model to infer.
```shell
python mm_infer.py
```

### Step5
Evaluate model on COCO dataset.
```shell
cd <path of yolov5>
python cnmm_val.py --weights MagicMind/python/yolov5_m_int8fp16.model--data data/coco.yaml --half --batch-size 1 --workers 1 --device cpu
```

### Step6
Profile MagicMind model.
```shell
cd MagicMind/python/
python mm_perf.py --mm_file_name yolov5_m_int8fp32.model
```
Use tensorboard to view profiling results.
```shell
tensorboard --port 8833 --logdir profile_data_output_dir/ --bind_all
```
