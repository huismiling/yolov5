from typing import List
import cv2
import numpy

import magicmind.python.runtime as mm
from magicmind.python.common.types import get_numpy_dtype_by_datatype

import os
import sys
sys.path.append("/workspace/zhangxiao/work/yolov5/")
from utils.datasets import LoadImages

class FixedCalibData(mm.CalibDataInterface):

    def __init__(self, shape: mm.Dims, data_type: mm.DataType, max_samples: int, data_paths: str):
        super().__init__()
        self.shape_ = shape
        self.data_type_ = data_type
        self.batch_size_ = shape.GetDimValue(0)
        self.input_wh = [shape.GetDimValue(3), shape.GetDimValue(2)]
        self.dataset = iter(LoadImages(data_paths, img_size=self.input_wh, stride=32, auto=False))
        self.max_samples_ = min(max_samples, len(self.dataset))

        self.current_sample_ = None
        self.outputed_sample_count = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return self.data_type_

    def get_sample(self):
        return self.current_sample_

    def __len__(self):
        return self.max_samples_

    def next(self):
        beg_ind = self.outputed_sample_count
        end_ind = self.outputed_sample_count + self.batch_size_
        if end_ind > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "End reached")

        imgs = []
        for it in range(self.batch_size_):
            img = next(self.dataset)[1]
            imgs.append(img[numpy.newaxis, :].astype(numpy.float32)/255.0)
        ret = numpy.concatenate(tuple(imgs), axis = 0)
        self.current_sample_ = numpy.ascontiguousarray(ret.astype(dtype = get_numpy_dtype_by_datatype(self.data_type_)))
        
        self.outputed_sample_count = end_ind
        return mm.Status.OK()

    def reset(self):
        self.current_sample_ = None
        self.outputed_sample_count = 0
        return mm.Status.OK()
