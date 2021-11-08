#include "infer_model.h"  // NOLINT
#include <memory>

using std::istringstream;

InferModel::InferModel(const char *model_name) {
  model_name = model_name;
  cnrtCreateQueue(&queue);
  // create model, enggine and context
  model = magicmind::CreateIModel();
  PTR_CHECK(model);
  MM_CHECK(model->DeserializeFromFile(model_name));

  engine = model->CreateIEngine();
  PTR_CHECK(engine);
  context = engine->CreateIContext();
  PTR_CHECK(context);
  auto input_dims = model->GetInputDimensions();
  // create and get irttensor from context
  MM_CHECK(CreateInputTensors(context, &input_tensors));
  MM_CHECK(CreateOutputTensors(context, &output_tensors));

  for (uint32_t i = 0; i < input_tensors.size(); ++i) {
    MM_CHECK(input_tensors[i]->SetDimensions(input_dims[i]));
  }
  MM_CHECK(context->InferOutputShape(input_tensors, output_tensors));

  // malloc host and device buffer and set ptr in tensor
  buffer = new MMBuffer();
  buffer->Init(input_tensors, output_tensors);
  
}

InferModel::~InferModel() {
  // free resources
  buffer->Destroy();
  for (auto tensor : input_tensors)
    tensor->Destroy();
  for (auto tensor : output_tensors)
    tensor->Destroy();
  cnrtDestroyQueue(queue);
  context->Destroy();
  MM_CHECK(engine->Destroy());
  model->Destroy();
}

void InferModel::infer(std::shared_ptr<float> in_ptr, std::shared_ptr<float> &out_ptr) {
  auto input_cpu_ptrs = buffer->InputCpuBuffers();
  auto output_cpu_ptrs = buffer->OutputCpuBuffers();

  memcpy(input_cpu_ptrs[0], (void*)(in_ptr.get()), input_tensors[0]->GetSize());
  // copy host inputdata to device
  buffer->H2D(queue);
  MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
  // copy device outputdata to host
  buffer->D2H(queue);
  CNRT_CHECK(cnrtSyncQueue(queue));
  memcpy(out_ptr.get(), output_cpu_ptrs[0], output_tensors[0]->GetSize());
}

