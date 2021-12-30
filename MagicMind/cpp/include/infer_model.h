/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef INFER_MODEL_H_
#define INFER_MODEL_H_

#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <string>
#include "cnrt.h"
#include "interface_runtime.h"
#include "sample_common.h"
#include "sample_buffer.h"

class InferModel {
  public:
    explicit InferModel(const char *model_name);
    ~InferModel();
    void infer(std::shared_ptr<float> in_ptr, std::shared_ptr<float> &out_ptr);
  private:
    std::string model_name;
    magicmind::IModel* model;
    cnrtQueue_t queue;
    magicmind::IEngine* engine;
    magicmind::IContext* context;
    MMBuffer *buffer;
    // std::vector<void *> input_cpu_ptrs;
    // std::vector<void *> output_cpu_ptrs;
    std::vector<magicmind::IRTTensor *> input_tensors;
    std::vector<magicmind::IRTTensor *> output_tensors;

};
#endif  // INFER_MODEL_H_

