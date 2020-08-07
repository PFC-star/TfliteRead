# Tflite在Mali GPU上运行过程介绍 - 以自带的Benchmark程序为例
Tflite自带的benchmark程序入口在`tensorflow/lite/tools/benchmark/benchmark_main.cc`文件中，其主要包括两个阶段：**初始化阶段**和**运行阶段**。其中前者包括读取`.tflite`文件并将之解析得到一个计算图，然后配置GPU生成kernel。而后者则是调用相关的Kernel完成网络的真正的推理计算。

## 1. 初始化阶段
### 1.1 解析传入参数及读取模型
由于这部分些代码并不属于关键内容，故而也没细看，此处只是简要介绍流程。进入`main()`函数程序首先调用`ParseFlags()`解析传入进来的命令行参数，比如输入模型文件的路径，是否采用GPU，是否使用OpenCL作为后端等，这些参数解析完成后会放在一个全局变量`params_`当中。而后，程序调用	`BuildFromFile()`函数根据传入的tflite模型路径，基于flatbuffer库读取文件，解析出模型，返回一个`unique_ptr<FlatBufferModel>`类型的变量`model_`。

### 1.2 Interpreter的初始化
在完成模型解析以及模型读取后，程序调用`InitInterpreter()`开始进行Interpreter的初始化。

#### **\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# 算子的注册 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

初始化的第一步是(CPU)算子的注册，程序中调用`GetOpResolver()->BuiltinOpResolver()`完成这个功能，这个函数中部分代码如下：

```C++
BuiltinOpResolver::BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_ABS, Register_ABS());
  AddBuiltin(BuiltinOperator_HARD_SWISH, Register_HARD_SWISH());
  // ...
  AddCustom("AudioSpectrogram", 
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());
}
```
其中`AddBuiltin()/AddCustom()`第一个参数是一个`enum int`，可以称之为op_code，标志着一个op，第二个参数是一个`TfliteRegistration`的对象，这个对象用几个函数指针描述了一个算子的`init`，`free`，`prepare`，`invoke`的具体实现(`tensorflow/lite/kernels/`底下维护了各个算子这几个函数的具体实现)。这个数据结构代码如下：

```C++
typedef struct TfLiteRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  const char* (*profiling_string)(const TfLiteContext* context, const TfLiteNode* node);
  int32_t builtin_code;
  const char* custom_name;
  int version;
} TfLiteRegistration;
```

`AddBuiltin()/AddCustom()`这个函数就是在向两个`unordered_map`类型的变量`builtins_/custom_ops_`中添加这两个`opcode->TfLiteRegistration`的映射，而这个变量维护在`BuiltinOpResolver`的基类`MutableOpResolver`中。在完成注册后，`GetOpResolver()`将这个`OpResolver`的指针返回，以用于后续构造`Interpreter`。

#### **\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# Interpreter的构建 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

首先我们在这里明确几个数据结构，其中只列出了部分比较关键的成员，可以在看下面程序流程时返回参考
```C++
typedef struct TfLiteTensor {
  // 标志dtype，可以为fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool等
  TfLiteType type;
  // 一个由fp16，fp32，fp64，uint8，int8，int16，int32
  // int64，bool等类型指针以及void*构成的Union
  TfLitePtrUnion data;
  // 用于记录tensor维度的类似于vector的结构体
  TfLiteIntArray* dims;
  // 标志内存分配策略
  TfLiteAllocationType allocation_type;
  // 存储这个Tensor所需要的字节数，等于dims[0]*...*dims[n]*sizeof(dtype)
  size_t bytes;
  const void* allocation;
  // 这个tensor的名字
  const char* name;
  // 标记这个Tensor将在什么delegate
  struct TfLiteDelegate* delegate;
  TfLiteBufferHandle buffer_handle;
  bool data_is_stale;
  // 记录这个Tensor是否是variable
  bool is_variable;
  // 量化相关参数
  TfLiteQuantization quantization;
  // 稀疏性相关参数
  TfLiteSparsity* sparsity;
  const TfLiteIntArray* dims_signature;
} TfLiteTensor;

typedef struct TfLiteNode {
  // 这个Node的input的node的index
  TfLiteIntArray* inputs;
  // 这个Node的output的node的index
  TfLiteIntArray* outputs;
  TfLiteIntArray* intermediates;  
  TfLiteIntArray* temporaries;
  // 指向一个OpData的结构体，其实op的init()函数的返回值，不同的算子维护的信息不同
  void* user_data;
  void* builtin_data;
  const void* custom_initial_data;
  int custom_initial_data_size;
  struct TfLiteDelegate* delegate;
} TfLiteNode;

typedef struct TfLiteContext {
  // Context中的tensor数量，其实就是subgraph以及interpreter
  // 的tensors vector的size()的值
  size_t tensors_size;
  // 指向subgraph以及interpreter的tensors vector中的data部分
  TfLiteTensor* tensors;
  // opaque full context ptr (an opaque c++ data structure)
  void* impl_;
  // 用于gemmlowp或者eigen的线程数设置
  int recommended_num_threads;
  // 允许用fp16代替fp32的标志位
  bool allow_fp32_relax_to_fp16;
} TfLiteContext;

class Subgraph {
 private:
  std::vector<TfLiteTensor> tensors_;
  TfLiteContext context_ = {};
  TfLiteExternalContext** external_contexts_;
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>> nodes_and_registration_;
  bool consistent_ = true;

  // input tensor在tensors_的下标
  std::vector<int> inputs_;
  // output index在tensors_的下标
  std::vector<int> outputs_;
  // 是variable的tensor在tensors_的下标
  std::vector<int> variables_;
  // The error reporter delegate that tflite will forward queries errors to.
  int next_execution_plan_index_to_prepare_;
  int next_execution_plan_index_to_plan_allocation_;
  // subgraph中所有node的一个op
  std::vector<int> execution_plan_;
  // 在apply delegate之前用于备份CPU版本的execution_plan_
  // 以用于设置delegate失败后restore
  std::vector<int> pre_delegation_execution_plan_;
  // 在这个subgraph上成功创建的delegate
  std::vector<TfLiteDelegate*> delegates_applied_;
  bool delegates_undone_ = false;
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> plan_cache_;
  std::vector<TfLiteDelegateParams> partitioning_preview_cache_;
  bool should_apply_nnapi_delegate_ = false;
  bool applied_nnapi_delegate_ = false;
  // 由于记录partition的结果
  std::vector<TfLiteDelegateParams> partitioning_preview_cache_;
  std::unique_ptr<MemoryPlanner> memory_planner_;
  bool tensor_resized_since_op_invoke_ = false;

  // 这个subgraphs_指针指向Interpreter::subgraphs_
  std::vector<std::unique_ptr<Subgraph>>* subgraphs_ = nullptr;
  bool has_dynamic_tensors_ = true;
  bool (*check_cancelled_func_)(void*) = nullptr;
  void* cancellation_data_ = nullptr;
  resource::ResourceMap* resources_ = nullptr;
};

class Interpreter {
 private:
  TfLiteContext* context_ = nullptr;
  std::vector<TfLiteDelegatePtr> owned_delegates_;
  TfLiteExternalContext* external_contexts_[kTfLiteMaxExternalContexts];
  std::unique_ptr<ExternalCpuBackendContext> own_external_cpu_backend_context_;
  // 这个interpreter所有的subgraphs，一般只有1个
  std::vector<std::unique_ptr<Subgraph>> subgraphs_;
  TfLiteDelegatePtr lazy_delegate_provider_;  
};

```

经过前面几步，我们已经获得了`FlatBufferModel`类型的模型变量`model_`，包含所有Op的`OpResolver`类型变量`resolver`，以及包含命令行参数的`params_`，现在程序将用这三个变量，对默认初始化的`Interpreter`变量`interpreter_`进行设置，这里是重载了`InterpreterBuilder`类的函数调用运算符。

```C++
tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, params_->num_threads); 
```

在这个函数里，其首先对传入的`params_`和`model_`做了一系列的检查，然后做了如下的事情：</br>
```C++
TfLiteStatus InterpreterBuilder::operator()(std::unique_ptr<Interpreter>* interpreter, int num_threads) {
  // 构建一个`model_`中各个op的`index->TfLiteRegistration`的映射
  // 存在`InterpreterBuilder::flatbuffer_op_index_to_registration_`中
  BuildLocalIndexToRegistrationMapping();
  // 设置`Interpreter::subgraphs_[i]->context()`中的
  // `recommend_thread`数为`params_`里的线程数
  (*interpreter)->SetNumThreads(num_threads);
  // 如果`model_`中的子图数大于1，则在`Interpreter::subgraphs_`
  // 中添加默认构造的新的子图
  (*interpreter)->AddSubgraphs(subgraphs->size() - 1);

  // 遍历`model_`的所有子图
  for (int subgraph_index = 0; subgraph_index < subgraphs->size(); ++subgraph_index) {
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index]; // `model_`的子图
    tflite::Subgraph* modified_subgraph = (*interpreter)->subgraph(subgraph_index); // `interpreter_的子图`
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();

    // 根据`model_->subgraphs[i]->tensors`数量来为
    // `interpreter_->subgraphs->tensors_`分配空间
    // 并将`tensors_->data`赋值给`subgraph->context_->tensors`
    modified_subgraph->AddTensors(tensors->size());
    // 把`model_->input`的index拷贝给`interpreter_->subgraph->input_`
    modified_subgraph->SetInputs(FlatBufferIntArrayToVector(subgraph->inputs()));
    // 把`model_->output`的index拷贝给`interpreter_->subgraph->output_`
    modified_subgraph->SetOutputs(FlatBufferIntArrayToVector(subgraph->outputs()));

    // 遍历`model_->operators`来初始化`interpreter_->subgraph->nodes_and_registration_`,
    // 具体来说，程序会读取各个算子的opcode判断这些算子属于什么类型，然后获取这些
    // 算子对应的`TfLiteRegistration`对象，再然后调用`ParseOpData()`解析这些算
    // 子，其底层分别调用各自对应的函数，比如`ParseConv2d`，`ParseDense`
    // 等等，解析出来的信息包括有stride，pad啊之类的东西，这些信息放在
    // `builtin_data`。最后调用`AddNodeWithParameters()`将解析出来的信息，
    // 以及输入输出参数整合成一个`pair<TfLiteNode, TfLiteRegistration>`，
    // 然后放进`interpreter_->subgraph->nodes_and_registration_`中，
    // 并将这个op的index push到`interpreter_->subgraphs->execution_plan_`中，
    // 由于此处的遍历算子是拓扑排序的，所以`execution_plan_`中的index就是
    // 拓扑排序的，这个函数在后面设置GPU Delegate时还会用到。
    ParseNodes(operators, modified_subgraph);

    // 遍历`model_->tensor`，获取算子的dim，dtype，tensor的大小
    // 及实际的data，**然后调用`quantization()`和`ParseQuantization()`
    // 获取并解析量化信息**，`sparsity()`和`ParseSparsity()`获取并解
    // 析稀疏信息，这些配置信息其实是在flatbuffers的schema文件中。最后
    // 调用`SetTensorParametersReadOnly()`将上面解析出来的各种信息来设置
    // `interpreter_->subgraph->context_->tensor`
    ParseTensors(buffers, tensors, modified_subgraph);

    // 遍历`interpreter_->subgraphs->tensors_`，找到是variable的那些tensor，
    // 将这些tensor的index存起来
    modified_subgraph->SetVariables(std::move(variables));
  }
  // 调用NNPACK的接口，没用到也没细看
  (*interpreter)->lazy_delegate_provider_ = MaybeCreateXNNPACKDelegate(num_threads);
  // 设置代理，把计算任务托付给具体硬件平台，但函数里面并没有执行，
  // 估计是配合上面NNPACK使用
  ApplyDelegates(interpreter->get(), num_threads);
}
```

在完成上述之后在有相关代码配置cache的使用，但这是针对于CPU的，所以此处没有细看。到此，Tflite在CPU上运行的设置其实已经基本完成了，可以看到此时subgraph中已经有了各个op(`nodes_and_registration_`)，tensor(`tensors_`)，以及遍历算子的顺序(`execution_plan_`)，后面的就是针对于GPU的配置。

### 1.3 GPU Delegate的设置

开始设置GPU代理，程序中会遍历所有的`delegate_provider`(COREML, Default-NoDelegate, EXTERNAL, GPU, Hexagon, NNAPI, XNNPACK七种provider以单例模式注册在系统中)，根据`param_`中维护的传入的delegate信息，初始化相应的delegate，此处就是GPU。

#### **\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# Delegate的创建 \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

程序中调用`CreateTfLiteDelegate()`获取一个`TFLiteDelegate`类型指针，这个指针指向一个`GPUDelegate`。

具体来说这个函数中首先调用`TfLiteGpuDelegateOptionsV2Default()`获取一个默认的GPU配置。然后根据传入的参数，修改这个默认GPU配置，比如是否允许精度损失，是否量化，cl还是gl等等。最后再调用`CreateGPUDelegate()->TfLiteGpuDelegateV2Create()`根据最终的GPU配置，创建GPU代理，即new了一个`tflite::gpu::Delegate()`。这个`tflite::gpu::Delegate`类有一个`struct TFLiteDelegate`类型的结构体，这个结构体里有个`data_`指针，这个指针就指向自己所属的这个`tflite::gpu::Delegate`对象，最后最终将这个`TFLiteDelegate`结构体的指针返回回去了。也不知道为啥要设计的这么绕。

#### **\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# 根据设置的Delegate更新subgraph \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

这一部分是程序将workload由CPU端转到GPU端的关键，入口函数是`interpreter_->ModifyGraphWithDelegate()`，其中调用每个subgraph的`ModifyGraphWithDelegate()`。

在这个函数中首先有个`RedoAllDelegates()`函数，重新re-apply所有的delegate，如果之前没有调用`UndoAllDelegate()`则不会执行，一般在配置Delegate失败的时候会出现这种情况，正常来说并不执行。

下一步程序调用`PrepareOpsStartingAt()`，这个函数里面以拓扑序遍历这个subgraph中的node，并调用这个node的`prepare()`函数，完成了各个算子的准备。这个准备其实就是对算子的输入输出做一些检查，并对算子的`OpData`结构体中一些配置进行了设置。

在之后调用`SwitchToDelegateContext()`，其实就是`subgraph->context_`几个函数指针的赋值，这几个函数指针在CPU版本本来是为空的，在Delegate的情况下却是需要的，所以此处做一个设置然后进行后续的设置。

经过简单设置后，最终程序调用`delegate->Prepare()`函数，这其实是个函数指针，其真实的函数是`tflite::gpu::Delegate().DelegatePrepare()`，这个函数完成了真实的workload delegate的工作，具体如下。

1. 创建了一个`TfLiteRegistration`对象，定义几个lambda赋值给这个对象的`init`，`free`，`prepare`，`invoke`四个函数指针实现，注意这四个函数指针，问就很重要。
2. 获取一下这个`tflite::gpu::Delegate`的指针
3. `GetOpsToReplace()` - 这个函数查找`subgraph->nodes_and_registrations_`中哪些op需要被replace成GPU版本。
    * 为了实现这个功能，该函数里定义了`GraphPartitionHelper`对象，然后调用其的`Partition()`函数对根据subgraph进行划分，得到一系列包含若干个算子的op的子集`NodeSubset`
    * 其划分的依据就是由若干个不支持的算子将原来的图分为几部分，具体来说，上面这个对象有两个变量，`supported_nodes_`和`unsupported_nodes_info`，分别表示在这个delegate上支持和不支持的那些算子信息。这些信息由`PrepareSupportedNodes()`函数遍历所有算子得到，其判断每个算子支持与否并记录在相关数据结构中。
    * 然后调用`PreviewDelegatePartitioning()->PartitionGraphIntoIndependentNodeSubsets()`对上面算子进行划分，因为测试用的mobilenet_v2所有算子GPU都支持，所以此处实际上将整张图变为一个Subset，这个subset中包含subgraph中所有的算子以及整图input和output的index。若有些不支持，其可能会根据不支持算子的位置，将网络划分为多个Subset，然后有些在GPU上执行，有些在CPU上执行
    * 这个函数同时还找到那些需要被replace的算子的index，并在最后会将之返回
    * 最后调用`PopulatePreviewDelegateParams()`将将划分的结果(比如delegate，)存在`subgraph->partitioning_preview_cache_`中
4. **`ReplaceNodeSubsetsWithDelegateKernels()`** 上面已经找到了知道了具体的delegate，需要替换的op的index，并且创建了一个`TfLiteRegistration`对象，现在程序中利用这三个信息来替换CPU node为GPU Kernel，并对应的修改了`execution_plan_`，由于内容太多，请参考下面这一小节。
5. 替换完成后这里会调用`SwitchToKernelContext()`，将`subgraph`的`context_`中部分函数指针设为delegation相关函数，然后释放一些不要的信息
6. 将设置好的这个delegate的指针存入`subgraph->delegates_applied_`这个vector中

#### **\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# CPU Node -> GPU Kernel \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

先关注几个数据结构，其中`NodeSubset`即上面创建的subset，而`TfLiteDelegateParams`则是用于解析这个`NodeSubset`并将之送给Delegate。
```C++
struct NodeSubset {
  // 这个Subset中的node的index
  std::vector<int> nodes;
  // 这个Subset中的input tensor的下标，一般来说是所有Node的
  // input tensor unique后的结果
  std::vector<int> input_tensors;
  // 这个Subset中的output tensor的下标
  std::vector<int> output_tensors;
};

typedef struct TfLiteDelegateParams {   
  // 指向Delegate对象的指针
  struct TfLiteDelegate* delegate;
  // 这三个参数和`NodeSubset`对应，用于利用`NodeSubset`创建delegate kernel
  TfLiteIntArray* nodes_to_replace;
  TfLiteIntArray* input_tensors;
  TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

```

承接上面的`ReplaceNodeSubsetsWithDelegateKernels()`，其核心代码如下，逻辑就是根据上面得到的信息(`node_subset`等)，为每个subset创建一个`BuiltinOperator_DELEGATE`的Node，然后将这个Node和上面创建的`TfLiteRegistration`对象构成一个pair然后放入`subgraph`的`nodes_and_registrations_`中，同时清空了原来CPU上的`execution_plan_`。
```C++
TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // 将上面的`TfLiteRegistration`对象的`builtin_code`设置为`BuiltinOperator_DELEGATE`
  registration.builtin_code = BuiltinOperator_DELEGATE; 
  // 清空了原来的execution_plan_
  execution_plan_.clear();

  int node_index;
  // 将`NodeSubset`中的信息解析到`TfLiteDelegateParams`类型变量中
  TfLiteDelegateParams* params = CreateDelegateParams(delegate, node_subset);
  // 创建一个node->registration的pair，push到`nodes_and_registration_`中
  AddNodeWithParameters(node_subset.input_tensors, node_subset.output_tensors, 
    {}, nullptr, 0, params, &registration, &node_index));

  TfLiteNode* node = &nodes_and_registration_[node_index].first;
  node->delegate = delegate;  //更新刚刚push的这个node的delegate
}
```

看起来很简单，但关键就是这个node的构建。其流程就是`AddNodeWithParameters()`函数中创建了一个`TfLiteNode`类型的变量`node`，然后将之`inputs`，`outputs`等设为`NodeSubset`的输入输出。最后再调用这个Node对应的`TfLiteRegistration`的`init()`函数，而这个函数就是之前的那个Lambda表达式。

```C++
[](TfLiteContext* context, const char* buffer, size_t) -> void* {
  // 上面从`NodeSubset`解析出来的信息
  const auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
  // 获取这个`Delegate`的指针
  auto* gpu_delegate = GetDelegate(params->delegate);
  auto gpu_delegate_kernel = absl::make_unique<DelegateKernel>(gpu_delegate);
  // 进行真正的Prepare
  const auto status = gpu_delegate_kernel->Prepare(context, params);
  return gpu_delegate_kernel.release();
}
```

可以看到这里init就是调用`gpu_delegate_kernel->Prepare(context, params)`，跳进去再细看

```C++
absl::Status Prepare(TfLiteContext* context,
                      const TfLiteDelegateParams* delegate_params) {
  GraphFloat32 graph;
  std::vector<uint32_t> input_refs, output_refs;
  InitializeGraph(context, delegate_params, &graph, &input_refs, &output_refs));

  std::unique_ptr<InferenceBuilder> builder;
  bool graph_is_destroyed;
  InitializeOpenClApi(&graph, &builder, &graph_is_destroyed);

  return builder->Build(&runner_);
}
```

可以看到这里创建了一个新Graph，并且其中数据是Float32的，然后基于原来的subgraph对之进行了设置，这里先看一下这个Graph的数据结构。

```C++
// tensor reference，其中并不维护data
template <typename ShapeT>
struct TensorRef {
  using ShapeType = ShapeT;
  DataType type = DataType::UNKNOWN;
  ShapeT shape;
  // 存着这个Tensor在subgraph的`tensors_`中的index
  int64_t ref = -1;
};

// graph中对tensor的封装，其中不维护data
struct Value {
  const ValueId id;
  TensorRef<BHWC> tensor;
  absl::optional<QuantizationParams> quant_params;
};

struct Operation {
  std::string type;
  absl::any attributes;
};

struct Node {
  const NodeId id;
  Operation operation;
};

class GraphFloat32 {
 private:
  // 对应subgraph中的node
  struct NodeDef {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::unique_ptr<Node> node;
  };
  // 对应subgraph中的tensor
  struct ValueDef {
    Node* producer = nullptr;
    std::vector<Node*> consumers;
    std::unique_ptr<Value> value;
  };
  std::vector<ValueDef> values_;
  std::map<NodeId, NodeDef> nodes_;
  // 这个Graph上的Node的拓扑排序
  std::vector<NodeId> execution_plan_;
};
```

再看看程序是如何对这个Graph进行设置的，其实调用`InitializeGraph()`，这个函数只是一层封装，里面其实是调用了`BuildFinalModel()`函数。在该函数里，其首先调用`BuildModel()`构建图，然后再调用`ApplyGeneralTransformations()`在这个图上做了一些优化。

```C++
absl::Status BuildModel(TfLiteContext* context,
                        const TfLiteDelegateParams* delegate_params,
                        GraphFloat32* graph,
                        std::unordered_map<int, int>* quant_conversion_map) {
  std::vector<std::unique_ptr<TFLiteOperationParser>> operations;
  std::vector<int> tflite_nodes;
  // 遍历所有的需要replace的op，获取其Node和Registration，
  // 并为之创建Parser（每种Op都实现了Parse()函数）
  for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
    GetNodeAndRegistration(context, delegate_params->nodes_to_replace->data[i], 
                           &tflite_node, &registration);
    auto op_parser = NewOperationParser(registration, quant_conversion_map != nullptr);
    operations.push_back(std::move(op_parser));
    tflite_nodes.push_back(i);
  }
  std::unordered_map<int, Value*> tensor_to_value;
  // 量化模型中的IOTensor在GPU上被转化为FP的Tensor，并将这些
  // Tensor及其对应的Value存入到graph中
  PrecreateIOTensors(context, graph, delegate_params->input_tensors,
                     quant_conversion_map, &tensor_to_value);
  PrecreateIOTensors(context, graph, delegate_params->output_tensors,
                     quant_conversion_map, &tensor_to_value);
  // 遍历之前定义的那些parser
  for (int i = 0; i < operations.size(); ++i) {
    GetNodeAndRegistration(context, delegate_params->nodes_to_replace->data[tflite_nodes[i]], 
                           &tflite_node, &registration));
    ObjectReader reader(graph, context, tflite_node, &tensor_to_value, quant_conversion_map);
    const auto status = operations[i]->Parse(tflite_node, registration, graph, &reader);
  }
}
```

在`BuildModel()`中，其遍历所有需要replace的算子，为之创建parser。

然后调用`PrecreateIOTensors()`函数根据原来`NodeSubset`中的inputs和outputs对应的`tensor`来构建fp32的`tensor`，而如果原来的tensor是quant的，则在此时会dequantize，并用一个map `quant_conversion_map`构建这两个tensor的映射。之后，程序会将新得到的fp32的`tensor`封装成`Value`，并存储在`graph`中，需要注意的是只有`NodeSubset->inputs`中的tensor属性为非constant的才会被创建。这样子，这个graph中就有了input和output。

当输入输出创建完成后，程序会利用上面创建的Parser的`Parse()`来解析原来`subgraph`中的op的信息用于构建这个新`graph`。其实就是获取一些信息诸如stride，pad等，然后设置这个node的输入输出`Value`，这里的`Value`都是对`TfliteTensor`的封装，并不直接存储数据。具体来说`value.tensor.ref`指向这个tensor在`subgraph.tensor_`的index，同时还创建了一个`unordered_map<int, Value*>`的tensor_idx到Value的映射。到此我们就将一个`NodeSubset`转为了一个`GraphFloat32`。与原来的`subgraph`相比，这个`GraphFloat32`的Node数并没有变，但`Value`数和原来的`Tensor`数不一样，`Value`数为`Node+1`，即认为每个`Node`输出一个`Value`，再加上图的输入的那个`Value`。

更进一步程序将调用`ApplyGeneralTransformations()`在这个图apply了一些pass，具体没细看。从结果来看，transform之后`graph`中每个正常节点后都跟了一个`quantize_and_dequantize`节点。`Node`数和`Value`数都增加了。然后再将这个图的`input`和`output`的reference返回了回去，这个ref之后用于面向user和面向engine的input/output tensor间的链接。

当这个图创建了之后，程序开始做OpenCL上的准备，这里先再看几个数据结构，这几个数据结构对OpenCL上的环境进行了封装，有点绕。

```C++
// InferenceEnvironment并非environment的派生类
// 其中主要维护device_id, cl_context, command_queue (api.h)
class InferenceEnvironmentImpl : public InferenceEnvironment {  
  struct InferenceEnvironmentOptions {
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLContext egl_context = EGL_NO_CONTEXT;
    absl::Span<const uint8_t> serialized_binary_cache;
  };
  const InferenceEnvironmentOptions options_;

  class Environment {
    CLDevice device_; // 对OpenCL定义的一些封装
    CLContext context_;
    CLCommandQueue queue_;
    ProfilingCommandQueue profiling_queue_;
    ProgramCache program_cache_;
  };
  Environment environment_;
  InferenceEnvironmentProperties properties_;
};

class InferenceBuilderImpl : public InferenceBuilder {
  class InferenceContext {
    // 一些针对OpenCL的封装，在不同的平台上设置的值不一样
    bool need_flush_ = false;
    bool flush_periodically_ = false;
    int flush_period_ = 1;
    bool need_manual_release_ = false;
    CLEvent prev_enqueue_start_point_;
    CalculationsPrecision precision_;
    TensorStorageType storage_type_;
    // Directly mapped nodes from graph, but some of them "inactive" due
    //  to fusion (inactive = fused).
    // Memory is allocated only once, in ConvertOperations, and is not modified
    //  anywhere.
    struct CLNode {
      std::vector<std::unique_ptr<GPUOperation>> operations;
      std::vector<ValueId> inputs;
      std::vector<ValueId> outputs;
      std::vector<int2> ranges;
      std::string name;
    };
    // 这个图中所有CLNode的数组
    std::vector<CLNode> nodes_;
    // 描述各个Tensor的id，shape，dtype，layout，storage_type的一个数据结构
    TensorReserver tensor_reserver_;
    // 维护分配的cl_mem
    std::vector<Buffer> shared_buffers_;
    std::vector<Tensor> shared_buffer_tensors_;  // use references to memory from shared_buffers_
    std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;
    std::map<ValueId, Tensor> strong_shape_tensors_;
    std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;
    // 输入的tensor的index
    std::vector<ValueId> input_ids_;
    // 输出的tensor的index
    std::vector<ValueId> output_ids_;
  };
  std::unique_ptr<InferenceContext> context_;
  std::unique_ptr<GlInteropFabric> gl_interop_fabric_;
  Environment* environment_;

  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  std::unique_ptr<TensorTieFactory> tie_factory_;
};

// 一个`DelegateKernel`就负责一个`NodeSubset`在GPU上的执行
class DelegateKernel {
  Delegate* const delegate_;  // 指向其对应的那个`Delegate`
  // `cl_environment_`指向自己的那个opencl 运行环境，其指向一个`InferenceEnvironmentImpl`类型的对象
  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;
  std::unique_ptr<InferenceRunner> runner_;
  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  // 这是一个quant tensor的index和dequant之后的float tensor的index之间的映射
  std::unordered_map<int, int> quant_conversion_map_;
  std::thread::id thread_id_prepare_;
  bool enforce_same_thread_ = false;
};
```
除了这几个对环境的封装之外，Tflite还定义了一套用于表示底层的计算图相关的数据结构，这些数据结构用于辅助`context_`中的`CLNode`类型的`node_`的构建，代码如下：

```C++
// CL中的Tensor描述子，表示dtype，storage_type，layout
struct TensorDescriptor : public GPUObjectDescriptor {
  DataType data_type = DataType::UNKNOWN;
  TensorStorageType storage_type = TensorStorageType::UNKNOWN;
  Layout layout = Layout::UNKNOWN;
}

// CL中算子的描述
class Tensor : public GPUObject {
  cl_mem memory_;
  cl_mem image_buffer_memory_;  // for TensorStorageType::IMAGE_BUFFER only
  bool memory_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
};

// 定义一个CL层面op的src和des的tensor的dtype，storage_type，layout
struct OperationDef {
  CalculationsPrecision precision;
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;
};

// 一个CL层面Operation的定义，CL的各个Kernel里都继承了这个类，又实现了一些自己的东西
class GPUOperation {
  OperationDef definition_;
  std::vector<Tensor*> src_;
  std::vector<Tensor*> dst_;
  Arguments args_;
  std::vector<ElementwiseOperation*> linked_operations_;
};

// 对GPUOperation的一份封装
struct GPUOperationWithRefs {
  std::unique_ptr<GPUOperation> operation;
  std::vector<int> input_ids;
  std::vector<int> output_ids;
};

// CL层面的子图
struct GPUOperationsSubgraph {
  // 这个子图中的GPUOperations
  std::vector<GPUOperationWithRefs> operations;
  // 这个子图中的tensors
  std::vector<std::pair<BHWC, TensorDescriptor>> new_tensors;
};
```

程序中是调用`InitializeOpenClApi()`函数进行OpenCL的初始化，其中首先调用`NewInferenceEnvironment()`创建一个默认的`InferenceEnvironment`，其实主要就是加载OpenCL Lib，获取设备Info，Context，配置CommandQueue，获取这个`cl_environment_`以及其的`properties`。

当`cl_enviroment_`确定之后，程序的下一个任务就是构建一个`InferenceBuilder`，代码中是根据delegation的options，graph来调用`cl::InferenceEnvironment::NewInferenceBuilder()`函数完成。该函数中创建了一个`InferenceBuilderImpl`类型的指针，并调用其`Initialize()`函数，用上面的options，graph，environment_来初始化。具体来说，`Initialize()`函数中将这些options转为一个`CreateInferenceInfo`类型的变量`create_info`中，然后调用`InitFromGraph()`函数来设置`InferenceBuilder`中`InferenceContext`类型的对象`context_`，其核心代码如下：

```C++
absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    Environment* env) {
  // 遍历graph，获得各个Tensor的id，shape，data_type，layout等，
  // 然后调用SelectBestStorageTypes()函数选择一个最优的存储方法，
  // 比如Buffer或者Image_Buffer等等，最后将这些信息打包一个
  // `TensorDescriptor`类型的对象，存储在在`TensorReserver`类型
  // 的对象tensor_reserver_中
  ReserveGraphTensors(create_info, creation_context, graph);
  // 根据传入进来的参数对这个inference_context进行一系列设置
  // ...

  // 将grpah的input和output的下标赋值给`context_`的`input_ids_`和`output_ids_`
  CopyInAndOutIds(graph);
  // 遍历输入Graph的所有Node，找到每个Node对应输入与输出，并将之组
  // 织成一个`OperationDef`类型的对象，这个对象指明了一个Node的输
  // 入与输出，然后再用这个对象构建`GPUOperationsSubgraph`类型的
  // 只包含一个Op的Subgraph。不同类型的op有着不同的创建方式，其把
  // 很多信息给了这个CL层面的graph抽象。再用这些Subgraph创建CLNode，
  // 再把这些CLNode放到`context_`中的一个`node_`中。
  ConvertOperations(creation_context, graph, create_info.hints);
  // 对上述的vector<CLNode>进行算子的合并，合并规则看起来是将conv2d
  // 和quantize可以合并，add和quantize可以合并，合并后一个`CLNode`
  // 中可以有多个`GPuOperation`
  Merge();
  // 为更新后的`CLNode`分配`cl_mem`，其中一些对齐啊什么的处理全在里面，之后需要细看
  AllocateMemory(env->device(), creation_context.context);
  // 将分配的内存跟`CLNode`的各个`GPUOperation`的`src_`和`dst_`绑定起来
  BindMemoryToOperations();
  // 调用每个`CLNode`的`GPUOperation`自己的`Compile()`函数，各个
  // 算子里面用代码生成的方式生成了各自的cl代码，然后生成Kernel，这
  // 个Kernel存在`GPUOperation`的`kernel_`元素中
  Compile(creation_context);
  // 调用每个`GPUOperation`自己的`Tune()`函数，主要就是获得一个比
  // 较优的WorkGroup，不同算子有自己不同的实现
  Tune(tuning_parameters);
}
```

到此，`InferenceBuilder`中`InferenceContext`类型的对象`context_`基本就是设置完全了。由于这些设置都是非常底层的接口，tflite调用`LinkTensors()`将inference engine的io tensor和面向用户的图的io tensor链接起来，并将这个链接设置为`builder_`的`vector<TensorTieDef>`的`inputs_`和`outputs_`。

最后再用这个设置好的`builder`来init一个`InferenceRunnerImpl`类型的runner，其实也是调用`LinkTensors()`做了一个`builder`和`runner`的输入输出的链接。最后承接上一小节的第5，6点，再将处理好的`delegate`指针赋值给`subgraph->owned_delegates_`，完成了这个GPU Delegate Kernel的设置。最后再将这个Kernel对应的`node_and_registration`放入`execution_plan_`中。

// *ConvertOperation() ~ Tune() need to be specified*

### 1.4 Tensor的check及内存分配

该部分其实是由`interpreter_->AllocateTensors()`作为入口函数，其中又调用了`subgraph->AllocateTensors()`，该函数内又调用了`PrepareOpsAndTensors()`函数为下一个要执行的Node(`next_execution_plan_index_to_prepare_`和`next_execution_plan_index_to_plan_allocation_`维护)分配内存，具体的分配工作由一个`MemoryPlaner`类型的对象`memory_planner_`完成。其中有两个函数`PlanAllocations()`和`ExecuteAllocations()`。前者遍历整个图指定了alloc和dealloc的计划，具体来说有两个变量，`alloc_node_`是一个 `vector<int>`，`alloc_node_[i]`表示每个`tensor[i]`由哪个Node算出来的，在这个Node上执行分配，后者则是表示在哪个Node执行之后这个Tensor就用不到了可以释放。但在GPU上，这里真正执行的只有网络输入输出的tensor的存储分配。此处不多细述。

// *To Be Specified*

## 2. 运行阶段

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
其中`kernel_`就是之前生成的OpenCL code编译成的kernel，`work_group_size_`是`Tune()`函数调出来的一个设置。而`DispatchImplicit()`是`CLCommandQueue`的一个函数，在这里完成一个Kernel的真正的运行。
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


