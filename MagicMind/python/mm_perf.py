import json
import numpy as np
import cv2
import argparse
import sys

import torch

import magicmind.python.runtime as mm
from magicmind.python.runtime import Context

from magicmind_model import MagicMindModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数，必选
    parser.add_argument("--builder_config", help="",type=str, default="builder_config.json")
    parser.add_argument("--mm_file_name", help="", type=str, default="yolox_m_quantized.model")
    parser.add_argument("--mm_dump", help="", action="store_true")
    parser.add_argument("--image_path", help="", type=str, default="../../../assets/dog.jpg")
    parser.add_argument("--input_shapes", help="", type=list, default=[[8, 3, 640, 640]])
    parser.add_argument("--input_dtypes", help="", type=list, default=["float32"])
    parser.add_argument("--with_p6", action="store_true", help="")

    args = parser.parse_args()

    mm_model = MagicMindModel(args.mm_file_name, mm_dump=args.mm_dump, device_id=0)

    input_shape = args.input_shapes[0]
    img = np.random.rand(*input_shape).astype(np.float32)
    # img = torch.from_numpy(img)
    logdir = "profile_data_output_dir"
    options = mm.ProfilerOptions(mm.HostTracerLevel.kCritical, mm.DeviceTracerLevel.kOn)
    profiler = mm.Profiler(options, logdir)
    profiler.start()
    for itd in range(10):
        profiler.step_begin(1)
        outputs = mm_model(img)
        profiler.step_end()
    profiler.stop()
