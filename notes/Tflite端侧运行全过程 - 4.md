# Tflite在端侧CPU/GPU上运行全过程（四）

本文以Tflite自带的benchmark程序为例，说明一个来自converter的`.tflite`文件使用Tflite框架在端侧CPU/GPU上运行起来的全过程。该程序入口函数在`tensorflow/lite/tools/benchmark/benchmark_main.cc`中。本文包含五小节，其中前四节属于运行之前的初始化阶段，第五节是真正的运行阶段，各小节内容如下：

第一节：介绍Tflite框架读取解析模型文件得到Inference Workload的过程

第二节：介绍Tflite如何通过GPU Delegate的设置将Workload迁移到GPU上的过程

第三节：CPU算子到GPU端kernel的流程

第四节：GPU算子生成的细节

第五节：运行阶段的框架如何完成真正的推理计算

本节为第四节

## 6. GPU Kernel Codegen及内存分配

上一节看到跟GPU Kernel生成密切相关的三个函数`ConvertOperations()`，`AllocateMemory()`，`Compile()`，先在对以Winograd算子对这三个函数做详细的分析。

### 6.1 `ConvertOperations()`

这个函数用于将一个CPU Node转为GPU Operation，但并不会真正编译Kernel。其中会遍历传入的Subgraph中的所有Node，构建一个`OperationDef`类型的对象描述每个Node的输入与输出。之后会用这个Node输入输这个对象，以及Node本身作为函数`GPUOperationFromNode()`的输入创建`GPUOperation`，之后再将这些GPU Operations封装成`CLNode`，完成了转换。

然后我们再看`GPUOperationFromNode()`函数，其代码如下，其中会读取这个Op的类型做一个switch-case，对于不同的类型，会调用不同的处理函数。此处我们看Winograd，其调用的是`WinogradFromNode()`函数。

```C++

absl::Status GPUOperationFromNode(const CreationContext& creation_context,
                                  const OperationDef& op_def, ModelHints hints,
                                  const std::vector<Value*>& inputs,
                                  const std::vector<Value*>& outputs,
                                  const Node& node,
                                  GPUOperationsSubgraph* gpu_subgraph) {
  std::unique_ptr<GPUOperation>* gpu_op = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
  auto op_type = OperationTypeFromString(node.operation.type);
  switch (op_type) {
    case OperationType::CONVOLUTION_2D: {
      WinogradFromNode(creation_context, op_def, hints, input_shape, output_shape, attr, gpu_subgraph);
    case OperationType::SQUARED_DIFF:
    // other case
  }
  return absl::OkStatus();
}
```

Tflite中Winograd实现的是F(4x4, 3x3)，即输入是一个个6x6的Tile，然后输出是一个个4x4的Tile。考虑Winograd的计算公式，`Y = A_t[GgG_t * BtdB]A`，其中g是权值，d是输入。Tflite将这个计算过程拆分成三个算子这三个算子都继承自`GPUOperation`，第一个算子`Winograd4x4To36`，其实际上执行是`V = BtdB`，将输入转为Winograd的形式。第二个算子执行的是`[U*V]`，其中`U`由离线的执行的`GgGt`得到，相当于对原始的权值做一个变换，然后用`Conv_1x1_buffer`算子在运行时完成`X = [U*V]`。最后一个算子`Winograd36To4x4`是`Y = At[X]A`将`X`变幻回去。

具体代码中，`WinogradFromNode()`首先调用`IsSuitableForWinograd4x4To6x6()`判断是否采用Winograd，然后创建了两个tensor作为这三个算子的中间变量。然后依次调用了`SelectWinograd4x4To36()`，`SelectConvolutionForWinograd()`，`SelectWinograd36To4x4()`三个函数创建三个Node。其中第一个和第三个函数执行流程很相似，都是在里面分别`new`了一个`Winograd4x4To36`/`Winograd36To4x4`算子，并为`Bt`/`At`分配`cl_mem`，此处不多细说。

比较有意思的是`SelectConvolutionForWinograd()`，其里面做了种种判断，最终选择了使用`Conv_buffer_1x1`算子，然后会调用`RearrangeWeightsToWinograd4x4To6x6Weights()`函数在CPU上计算`U = GgG_t`，计算完成后其也会为之分配`cl_mem`，而在分配之前，这里调用了一个函数`RearrangeWeightsToOHWIOGroupI4O4()`，将权值组织成`OC/4, H, W, IC/4, IC4, OC4`的格式，在纸上画出了其内存排布后发现这样是为了输出也是C4的，此处也不多赘述了。

到此三个GPUOperation也已经创建完毕了。

### 6.2 `AllocateMemory()`

这里想说的其实是两件事，一个是Tflite中如何对`cl_mem`进行封装，另一个是Tflite中的内存复用策略，先看第一个。tflite中创建了两个类，分别表示opencl中的`buffer`和`image2d`，代码如下：

```C++
class Buffer : public GPUObject {
  cl_mem buffer_ = nullptr;
  size_t size_;
};
class Texture2D : public GPUObject {
  cl_mem texture_ = nullptr;
  int width_;
  int height_;
  cl_channel_type channel_type_;
};
```

然后又定义了一个`LinearStorage`，其描述一个`image2d`或者`buffer`，代码如下：

```C++
class LinearStorage : public GPUObject {
  Texture2D texture_storage_;
  Buffer buffer_storage_;
  cl_mem memory_ = nullptr;  // Just a reference to texture_storage_ or
                             // buffer_storage_ memory, not an owner
  int depth_;
  std::string name_;
  LinearStorageType storage_type_;  // enum类型 { BUFFER, TEXTURE_2D }
  DataType data_type_;				// dtype，fp32或者fp16
};
```

Tflite主要操作的对象`LinearStorage`，其执行到底层后会做判断，看是`buffer`还是`image2d`，然后再调用真正的opencl函数完成对`cl_mem`的操作。

好，第一件事到此就说完了，再看Tflite中的内存复用策略。

其实`AllocateMemory()`的主要工作就是算出一个合理的内存复用策略，然后进行实际的内存分配。之后还有个`BindMemoryToOperations()`完成分配好的内存与各个tensor的bind。而`AllocateMemory()`中主要是调用`AllocateMemoryForBuffers()`和`AllocateMemoryForStrongShapes()`，两者类似，现在主要看前者。

`AllocateMemoryForBuffers()`中首先找到那些使用buffer的tensor，将之加入buffer_usages中，usage是一个 `tensorIdx --> (startOpIdx, endOpIdx)`的`map`，表示这个`tensor`是最先由`startOpIdx`用到，最后由`endOpIdx`用到。这样我们可以知道各个tensor的生存期，之后程序再获取每个`Tensor`所需的Memory的大小，这样子就为内存复用优化创造了可能。之后程序中调用`AssignObjectsToTensors()`，看其代码

```C++
template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<size_t>* assignment,
    const UsageGraph* reallocation_graph) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignmentWithHash(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignment(usage_records, assignment, reallocation_graph);
    case MemoryStrategy::GREEDY_BY_BREADTH:
      return GreedyByBreadthAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BY_SIZE:
      return GreedyBySizeDistPriorityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BEST:
      return BestGreedy(usage_records, assignment);
    case MemoryStrategy::MINCOSTFLOW:
      return MinCostFlowAssignment(usage_records, assignment);
    default:
      return absl::InternalError("MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}
```

不同的策略对应着不同的分配方式，当确定了最终的分配方式后，之后会调用opencl相关函数完成内存分配。

### 6.3 `Compile()`

其实际上是一个`Codegen`生成Opencl的kernel代码，然后编译这个kernel的过程，对应不同`GPUOperation`有着不同的实现。以Winograd为例，如下，具体也不想细写了，主要就是有个代码模板，然后在对代码模板中的一些变量进行替换，得到cl的kernel，之后再调用`GetOrCreateCLKernel()`函数编译kernel。

```C++
absl::Status Winograd36To4x4::Compile(const CreationContext& creation_context) {
  std::string code = GetWinograd36To4x4Code(definition_, &args_);
  std::string element_wise_code; 
  MergeOperations(linked_operations_, &args_, &element_wise_code));
  args_.TransformToCLCode(creation_context.device->GetInfo(), {{"dst_tensor", element_wise_code}}, &code));
  creation_context.cache->GetOrCreateCLKernel(code, "main_function", options, 
      *creation_context.context, *creation_context.device, &kernel_));
}
```

