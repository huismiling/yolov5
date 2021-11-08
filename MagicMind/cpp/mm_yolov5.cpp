// This file is wirtten base on the following file:
// https://github.com/Tencent/ncnn/blob/master/examples/yolov5.cpp
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
// ------------------------------------------------------------------------------
// Copyright (C) [2020-2023] by Cambricon, Inc.

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include "cnrt.h"
#include "interface_builder.h"
#include "interface_network.h"
#include "interface_runtime.h"
#include "interface_profiler.h"
#include "infer_model.h"


#define YOLOV5_FEATURE_NUM 85   // class number
#define YOLOV5_ANCHORS_NUM 25200
#define YOLOv5_OUTPUT_NUM  (1*YOLOV5_ANCHORS_NUM*YOLOV5_FEATURE_NUM)   // output element number

#define YOLOV5_V60 1 //YOLOv5 v6.0

#if YOLOV5_V60
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
#endif //YOLOV5_V60

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const float* feat_ptr, float prob_threshold, std::vector<Object>& objects)
{

    const int num_class = YOLOV5_FEATURE_NUM - 5;

    for (int q = 0; q < YOLOV5_ANCHORS_NUM; q++)
    {
        const float* feat = feat_ptr + q*YOLOV5_FEATURE_NUM;

        // find class index with max class score
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float score = feat[5 + k];
            if (score > class_score)
            {
                class_index = k;
                class_score = score;
            }
        }

        float box_score = feat[4];

        float confidence = box_score * class_score;

        if (confidence >= prob_threshold)
        {
            // yolov5/models/yolo.py Detect forward
            // y = x[i].sigmoid()
            // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            float pb_cx = feat[0];
            float pb_cy = feat[1];

            float pb_w = feat[2];
            float pb_h = feat[3];

            float x0 = pb_cx - pb_w * 0.5f;
            float y0 = pb_cy - pb_h * 0.5f;
            float x1 = pb_cx + pb_w * 0.5f;
            float y1 = pb_cy + pb_h * 0.5f;

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = class_index;
            obj.prob = confidence;

            objects.push_back(obj);
        }
    }
}

static int detect_yolov5(const char *model_name, const cv::Mat& bgr, std::vector<Object>& objects)
{
    InferModel* mm_model = new InferModel(model_name);

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    cv::Mat in; //= ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);
    cv::resize(bgr, in, cv::Size(w, h));

    // // // pad to YOLOX_TARGET_SIZE rectangle
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    // // different from yolov5, yolox only pad on bottom and right side,
    // // which means users don't need to extra padding info to decode boxes coordinate.
    cv::Mat in_u8(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    in.copyTo(in_u8(cv::Rect(wpad/2, hpad/2, in.cols, in.rows)));

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8/16/32
    {
        size_t channels = in_u8.channels();
        size_t img_h = in_u8.rows;
        size_t img_w = in_u8.cols;

        std::shared_ptr<float> in_ptr(new float[channels*img_h*img_w]);
        std::shared_ptr<float> out_ptr(new float[YOLOv5_OUTPUT_NUM]);
        float* blob_data = in_ptr.get();
        for (size_t c = 0; c < channels; c++) 
        {
            for (size_t  h = 0; h < img_h; h++) 
            {
                for (size_t w = 0; w < img_w; w++) 
                {
                    blob_data[c * img_w * img_h + h * img_w + w] =
                        (float)(in_u8.at<cv::Vec3b>(h, w)[c]) * 1/255.0;
                }
            }
        }
        mm_model->infer(in_ptr, out_ptr);

        std::vector<Object> objects8;
        generate_proposals(out_ptr.get(), prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imwrite("image_out.jpg", image);
    // cv::imshow("image", image);
    // cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        LOGINFO("Usage: %s model_path image_path\n", argv[0]);
        LOGINFO("\nex. mm_yolov5 model image_path\n");
        return -1;
    }

    const char *model_name = argv[1];
    const char* imagepath = argv[2];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov5(model_name, m, objects);

    draw_objects(m, objects);

    return 0;
}