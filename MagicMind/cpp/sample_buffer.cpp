


#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <string>
#include "sample_buffer.h"

void MMBuffer::MallocBuffers(std::vector<magicmind::IRTTensor *> &tensors,
                             std::vector<void *> &cpu_ptrs,
                             std::vector<void *> &mlu_ptrs) {
  for (uint32_t i = 0; i < tensors.size(); ++i) {
    // malloc cpu ptr
    cpu_ptrs[i] = (void *)malloc(tensors[i]->GetSize());
    if (tensors[i]->GetMemoryLocation() == magicmind::TensorLocation::kHost) {
      // h2d in magicmind
      MM_CHECK(tensors[i]->SetData(cpu_ptrs[i]));
    } else if (tensors[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      // malloc mlu ptr
      CNRT_CHECK(cnrtMalloc(&(mlu_ptrs[i]), tensors[i]->GetSize()));
      MM_CHECK(tensors[i]->SetData(mlu_ptrs[i]));
    } else {
      LOGINFO("Not support dev type.");
    }
  }
}

// Malloc cpu and mlu buffer
bool MMBuffer::Init(std::vector<magicmind::IRTTensor *> &input_tensors,
                    std::vector<magicmind::IRTTensor *> &output_tensors) {
  input_num_  = input_tensors.size();
  output_num_ = output_tensors.size();
  if (input_num_ == 0 || output_num_ == 0) {
    return false;
  }
  input_tensors_  = input_tensors;
  output_tensors_ = output_tensors;
  mlu_input_ptrs_.resize(input_num_);
  cpu_input_ptrs_.resize(input_num_);
  mlu_output_ptrs_.resize(output_num_);
  cpu_output_ptrs_.resize(output_num_);

  MallocBuffers(input_tensors_, cpu_input_ptrs_, mlu_input_ptrs_);
  MallocBuffers(output_tensors_, cpu_output_ptrs_, mlu_output_ptrs_);

  return true;
}

bool MMBuffer::Init(std::vector<magicmind::IRTTensor *> &input_tensors) {
  input_num_  = input_tensors.size();
  if (input_num_ == 0) {
    return false;
  }
  input_tensors_  = input_tensors;
  mlu_input_ptrs_.resize(input_num_);
  cpu_input_ptrs_.resize(input_num_);

  MallocBuffers(input_tensors_, cpu_input_ptrs_, mlu_input_ptrs_);
  return true;
}


// if tensor location is mlu, memcpy host to device before enqueue [async]
void MMBuffer::H2D(cnrtQueue_t queue) {
  for (uint32_t i = 0; i < input_num_; ++i) {
    if (input_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpyAsync(mlu_input_ptrs_[i], cpu_input_ptrs_[i],
                                 input_tensors_[i]->GetSize(), queue, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
  }
}

// if tensor location is mlu, memcpy host to device before enqueue [sync]
void MMBuffer::H2D() {
  for (uint32_t i = 0; i < input_num_; ++i) {
    if (input_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpy(mlu_input_ptrs_[i], cpu_input_ptrs_[i], input_tensors_[i]->GetSize(),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
  }
}

// if tensor location is mlu, memcpy device to host after enqueue [async]
void MMBuffer::D2H(cnrtQueue_t queue) {
  for (uint32_t i = 0; i < output_num_; ++i) {
    if (output_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpyAsync(cpu_output_ptrs_[i], mlu_output_ptrs_[i],
                                 output_tensors_[i]->GetSize(), queue,
                                 CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
  }
}

// if tensor location is mlu, memcpy device to host after enqueue [sync]
void MMBuffer::D2H() {
  for (uint32_t i = 0; i < output_num_; ++i) {
    if (output_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpy(cpu_output_ptrs_[i], mlu_output_ptrs_[i], output_tensors_[i]->GetSize(),
                            CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
  }
}

void MMBuffer::Destroy() {
  for (uint32_t i = 0; i < input_num_; i++) {
    if (mlu_input_ptrs_[i]) {
      CNRT_CHECK(cnrtFree(mlu_input_ptrs_[i]));
      mlu_input_ptrs_[i] = nullptr;
    }
    if (cpu_input_ptrs_[i]) {
      free(cpu_input_ptrs_[i]);
      cpu_input_ptrs_[i] = nullptr;
    }
  }
  for (uint32_t i = 0; i < output_num_; i++) {
    if (mlu_output_ptrs_[i]) {
      CNRT_CHECK(cnrtFree(mlu_output_ptrs_[i]));
      mlu_output_ptrs_[i] = nullptr;
    }
    if (cpu_output_ptrs_[i]) {
      free(cpu_output_ptrs_[i]);
      cpu_output_ptrs_[i] = nullptr;
    }
  }
  delete this;
}

std::string MMBuffer::DebugString() {
  std::stringstream ret;
  ret << "Buffer Info:\n";
  ret << "InputNum: " << input_num_ << "\n";
  ret << "OutputNum: " << output_num_ << "\n";
  for (uint32_t i = 0; i < input_num_; i++) {
    ret << "Input[" << i << "]: \n";
    ret << "  Name: " << input_tensors_[i]->GetName() << "\n";
    ret << "  Datatype: " << magicmind::TypeEnumToString(input_tensors_[i]->GetDataType()) << "\n";
    ret << "  Layout: " << magicmind::LayoutEnumToString(input_tensors_[i]->GetLayout()) << "\n";
    ret << "  Dim: " << input_tensors_[i]->GetDimensions() << "\n";
    ret << "  Size: " << input_tensors_[i]->GetSize() << "\n";
    ret << "  Ptr Addr: " << input_tensors_[i]->GetMutableData() << "\n";
    ret << "  TensorLoc: " << TensorLocationEnumToString(input_tensors_[i]->GetMemoryLocation())
        << "\n";
  }
  for (uint32_t i = 0; i < output_num_; i++) {
    ret << "Output[" << i << "]: \n";
    ret << "  Name: " << output_tensors_[i]->GetName() << "\n";
    ret << "  Datatype: " << magicmind::TypeEnumToString(output_tensors_[i]->GetDataType()) << "\n";
    ret << "  Layout: " << magicmind::LayoutEnumToString(output_tensors_[i]->GetLayout()) << "\n";
    ret << "  Dim: " << output_tensors_[i]->GetDimensions() << "\n";
    ret << "  Size: " << output_tensors_[i]->GetSize() << "\n";
    ret << "  Ptr Addr: " << output_tensors_[i]->GetMutableData() << "\n";
    ret << "  TensorLoc: " << TensorLocationEnumToString(output_tensors_[i]->GetMemoryLocation())
        << "\n";
  }
  return ret.str();
}