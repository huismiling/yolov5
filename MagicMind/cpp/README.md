# YOLOX-CPP-MagicMind

## Tutorial

### Step1
Install MagicMind relative software and python wheel. Please following the reference https://developer.cambricon.com.

### Step2
Use Python generate magicmind model. Please following the python demo of MagicMind.

### Step3
Build mm_yolov5.
```shell
cd MagicMind/cpp/
mkdir build/
cd build/
cmake ..
make
```

### Step4
infer with MagicMind model.
```shell
./mm_yolox ../../python/yolov5_m_fp32.model ../../../data/images/bus.jpg
```

