# Tflite在端侧CPU/GPU上运行全过程（五）

本文以Tflite自带的benchmark程序为例，说明一个来自converter的`.tflite`文件使用Tflite框架在端侧CPU/GPU上运行起来的全过程。该程序入口函数在`tensorflow/lite/tools/benchmark/benchmark_main.cc`中。本文包含五小节，其中前四节属于运行之前的初始化阶段，第五节是真正的运行阶段，各小节内容如下：

第一节：介绍Tflite框架读取解析模型文件得到Inference Workload的过程

第二节：介绍Tflite如何通过GPU Delegate的设置将Workload迁移到GPU上的过程

第三节：CPU算子到GPU端kernel的流程

第四节：GPU算子生成的细节

第五节：运行阶段的框架如何完成真正的推理计算

本节为第五节

## 7. 推理运行阶段

当这些初始化的工作弄清楚之后，后面的执行部分就很简单了。在这个程序中其对运行做了很多层封装，`Run() -> RunImpl() -> interpreter_.Invoke() -> subgraph.Invoke()`，其核心代码如下：

```C++
TfLiteStatus Subgraph::Invoke() {
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
      
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;

    // 即考虑到存在不同算子在不同Delegate上, 这时需要确认不同算子之间是否需要Copy数据
    // .. Some Data Copy Operation
    OpInvoke(registration, &node);
  }
}
```

其核心逻辑就是遍历那个`execution_plan_`，执行每个execution plan的对应的那个node的`Invoke()`函数。而在初始化阶段，设置Delegate之后，`execution_plan_`只有一个`builtin_code`为`BuiltinOperator_DELEGATE`的node，即调用这个node的`invoke()`。这个`invoke()`之前由一个lambda赋值。

```C++
[](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
  const auto status = GetDelegateKernel(node)->Invoke(context);
  // error check code
}
```

这里其获取了这个node对应的`DelegateKernel`的指针，然后调用其`Invoke()`函数。再看这个`Invoke()`函数，其逻辑也很简单，先对输入进行Dequant，然后将这个输入设置给`InferenceContext`，然后调用`InferenceRunnerImpl::Run()`完成网络真正的运行，再将结果量化回去。

```C++
absl::Status Invoke(TfLiteContext* context) {
  const bool is_dequant_required = !quant_conversion_map_.empty();
  if (is_dequant_required) 
        DequantizeInputs(context, input_indices_, quant_conversion_map_);

  SetInputsAndOutputs(context);
  runner_->Run();
  if (is_dequant_required)
        QuantizeOutputs(context, output_indices_, quant_conversion_map_));
}
```

再看这个`Run()`函数，其完成从外部面向用户的input到面向inference engine的输入的拷贝，然后调用`InferenceContext::AddToQueue()`进行网络真正的运行，最后再将结果copy回用户空间。

```C++
absl::Status Run() override {
  for (auto& obj : inputs_) 
    RETURN_IF_ERROR(obj->CopyFromExternalObject());
  RETURN_IF_ERROR(context_->AddToQueue(queue_));
  clFlush(queue_->queue());
  for (auto& obj : outputs_) 
    RETURN_IF_ERROR(obj->CopyToExternalObject());
  return absl::OkStatus();
}
```

再看这个`AddToQueue()`函数，其遍历了所有`CLNode`，调用这些`Node`的`operations`的`AddToQueue()`

```C++
absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue) {
  if (need_manual_release_) {
    if (prev_enqueue_start_point_.is_valid()) {
      prev_enqueue_start_point_.Wait();
    }
    RETURN_IF_ERROR(queue->EnqueueEvent(&prev_enqueue_start_point_));
  }
  int counter = 0;
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operations[0]->AddToQueue(queue));
    counter++;
    if (flush_periodically_ && counter % flush_period_ == 0) {
      clFlush(queue->queue());
    }
  }
  if (need_flush_) {
    clFlush(queue->queue());
  }
}
```

而其实每个GPU Kernel都自己实现了这个`AddToQueue()`，这里以`depthwise_conv`为例，

```C++
absl::Status DepthwiseConvolution::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}
```

其中`kernel_`就是之前生成的OpenCL code编译成的kernel，`BindArguments()`主要就是为opencl kernel设置输入，`work_group_size_`是`Tune()`函数调出来的一个设置。而`DispatchImplicit()`是`CLCommandQueue`的一个函数，在这里完成一个Kernel的真正的运行。

```C++
absl::Status CLCommandQueue::DispatchImplicit(const CLKernel& kernel, int3 grid,
                                              int3 work_group_size,
                                              CLEvent* event) {
  std::vector<size_t> local(3);
  std::vector<size_t> global(3);
  for (int i = 0; i < 3; ++i) {
    local[i] = work_group_size[i];
    global[i] = AlignByN(grid[i], work_group_size[i]);
  }
  cl_event resulting_event;
  const int error_code = clEnqueueNDRangeKernel(
      queue_, kernel.kernel(), 3, nullptr, global.data(), local.data(), 0,
      nullptr, event ? &resulting_event : nullptr);
  if (event) {
    *event = CLEvent(resulting_event);
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clEnqueueNDRangeKernel - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}
```

