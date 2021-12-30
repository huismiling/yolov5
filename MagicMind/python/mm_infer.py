import json
import numpy as np
import cv2
import argparse
import sys
from mm_utils import COCO_CLASSES

import torch

import magicmind.python.runtime as mm
from magicmind.python.runtime import Context

from magicmind_model import MagicMindModel

sys.path.append("/workspace/zhangxiao/work/yolov5/")
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument("--mm_file_name", help="", type=str, default="yolov5_m_int8fp16.model")
    parser.add_argument("--mm_dump", help="", action="store_true")
    parser.add_argument("--image_path", help="", type=str, default="../../data/images/bus.jpg")
    parser.add_argument("--out_path", help="", type=str, default="mm_out.jpg")
    parser.add_argument("--input_shapes", help="", type=list, default=[[1, 3, 640, 640]])
    parser.add_argument("--input_dtypes", help="", type=list, default=["float32"])
    parser.add_argument("--with_p6", action="store_true", help="")

    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    args = parser.parse_args()
    stride, names = 64, COCO_CLASSES  # assign defaults

    mm_model = MagicMindModel(args.mm_file_name, mm_dump=args.mm_dump, device_id=0)

    imgsz = args.input_shapes[0][2:4]
    dataset = iter(LoadImages(args.image_path, img_size=imgsz, stride=stride, auto=False))
    path, img, im0s, vid_cap = next(dataset)

    img = img[np.newaxis, ].astype(np.float32)/255.0
    # img = torch.from_numpy(img)
    pred = mm_model(img)
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, 
                               args.agnostic_nms, max_det=args.max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        save_path = args.out_path  # img.jpg
        s += '%gx%g ' % img.shape[2:]  # print string
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        cv2.imwrite(save_path, im0)
