# Tflite在端侧CPU/GPU上运行全过程（三）

本文以Tflite自带的benchmark程序为例，说明一个来自converter的`.tflite`文件使用Tflite框架在端侧CPU/GPU上运行起来的全过程。该程序入口函数在`tensorflow/lite/tools/benchmark/benchmark_main.cc`中。本文包含五小节，其中前四节属于运行之前的初始化阶段，第五节是真正的运行阶段，各小节内容如下：

第一节：介绍Tflite框架读取解析模型文件得到Inference Workload的过程

第二节：介绍Tflite如何通过GPU Delegate的设置将Workload迁移到GPU上的过程

第三节：CPU算子到GPU端kernel的流程

第四节：GPU算子生成的细节

第五节：运行阶段的框架如何完成真正的推理计算

本节为第三节

## 5. CPU Node --> GPU Kernel

上面我们知道原始的subgraph先在被划分为若干个`Subset`，其会被封装成`TfLiteDelegateParams`并将之送给Delegate，这个类数据结构如下：
```C++
typedef struct TfLiteDelegateParams {   
  struct TfLiteDelegate* delegate;   // 指向Delegate对象的指针
  // 这三个参数和`NodeSubset`对应，用于利用`NodeSubset`创建delegate kernel
  TfLiteIntArray* nodes_to_replace;
  TfLiteIntArray* input_tensors;
  TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

```

承接上面的`ReplaceNodeSubsetsWithDelegateKernels()`，其核心代码如下，逻辑就是根据上面得到的信息(`node_subset`等)，将这个`Subset`封装成`TfLiteDelegateParams`，然后根据这个参数为每个subset创建一个`BuiltinOperator_DELEGATE`的Node，然后将这个Node和上面创建的`TfLiteRegistration`对象构成一个pair然后放入`subgraph`的`nodes_and_registrations_`中，同时清空了原来CPU上的`execution_plan_`。
```C++
TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // 将上面的`TfLiteRegistration`对象的`builtin_code`设置为`BuiltinOperator_DELEGATE`
  registration.builtin_code = BuiltinOperator_DELEGATE; 
  execution_plan_.clear();       // 清空了原来的execution_plan_

  int node_index;
  // 将`NodeSubset`中的信息解析到`TfLiteDelegateParams`类型变量中
  TfLiteDelegateParams* params = CreateDelegateParams(delegate, node_subset);
  // 创建一个node->registration的pair，push到`nodes_and_registration_`中
  AddNodeWithParameters(node_subset.input_tensors, node_subset.output_tensors, 
    {}, nullptr, 0, params, &registration, &node_index));

  TfLiteNode* node = &nodes_and_registration_[node_index].first;
  node->delegate = delegate;     //更新刚刚push的这个node的delegate
}
```

看起来很简单，但关键就是这个node的构建。其流程就是`AddNodeWithParameters()`函数中创建了一个`TfLiteNode`类型的变量`node`，然后将之`inputs`，`outputs`等设为`NodeSubset`的输入输出。最后再调用这个Node对应的`TfLiteRegistration`的`init()`函数，而这个函数就是之前的那个Lambda表达式。

```C++
[](TfLiteContext* context, const char* buffer, size_t) -> void* {
  // 上面从`NodeSubset`解析出来的信息
  const auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
  auto* gpu_delegate = GetDelegate(params->delegate);                  // 获取这个`Delegate`的指针
  auto gpu_delegate_kernel = absl::make_unique<DelegateKernel>(gpu_delegate);
  const auto status = gpu_delegate_kernel->Prepare(context, params);   // 进行真正的Prepare
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

可以看到这里Initialize一个新Graph，并且其中数据是Float32的，然后基于原来的subgraph对之进行了设置，这里先看一下这个Graph的数据结构。

```C++
template <typename ShapeT>    // tensor reference，其中并不维护data
struct TensorRef {
  using ShapeType = ShapeT;
  DataType type = DataType::UNKNOWN;
  ShapeT shape;  
  int64_t ref = -1;           // 存着这个Tensor在subgraph的`tensors_`中的index
};

struct Value {                // graph中对tensor的封装，其中不维护data
  const ValueId id;
  TensorRef<BHWC> tensor;
  absl::optional<QuantizationParams> quant_params;
};

struct Operation {
  std::string type;
  absl::any attributes;
};

struct Node {                // Graph中对Op的封装
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
  std::vector<ValueDef> values_;		// 这个graph中的tensors
  std::map<NodeId, NodeDef> nodes_;     // 这个graph中的ops
  std::vector<NodeId> execution_plan_;  // 这个Graph上的Node的拓扑排序
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
  };
  const InferenceEnvironmentOptions options_;

  class Environment {
    CLDevice device_; // 对OpenCL定义的一些封装
    CLContext context_;
    CLCommandQueue queue_;
  };
  Environment environment_;
  InferenceEnvironmentProperties properties_;
};

class InferenceBuilderImpl : public InferenceBuilder {
  class InferenceContext {  // 一些针对OpenCL的封装，在不同的平台上设置的值不一样
    CalculationsPrecision precision_;
    TensorStorageType storage_type_;
    struct CLNode {         // tflite对GPU Kernel的最终的封装
      std::vector<std::unique_ptr<GPUOperation>> operations;  // 实际的Kernel
      std::vector<ValueId> inputs;	// kernel的输入输出
      std::vector<ValueId> outputs;
      std::string name;
    };
    std::vector<CLNode> nodes_;       // 这个图中所有CLNode的数组
    TensorReserver tensor_reserver_;  // 描述各个Tensor的id，shape，dtype，layout，storage_type的一个数据结构
    std::vector<Buffer> shared_buffers_;         // 维护分配的cl_mem
    std::vector<Tensor> shared_buffer_tensors_;  // use references to memory from shared_buffers_
    std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;
    std::map<ValueId, Tensor> strong_shape_tensors_;
    std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;
    std::vector<ValueId> input_ids_;  // 输入的tensor的index
    std::vector<ValueId> output_ids_; // 输出的tensor的index
  };
  std::unique_ptr<InferenceContext> context_;
  Environment* environment_;
};

class DelegateKernel {        // 一个`DelegateKernel`就负责一个`NodeSubset`在GPU上的执行
  Delegate* const delegate_;  // 指向其对应的那个`Delegate`
  // `cl_environment_`指向自己的那个opencl 运行环境，其指向一个`InferenceEnvironmentImpl`类型的对象
  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;
  std::unique_ptr<InferenceRunner> runner_;
  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  // 这是一个quant tensor的index和dequant之后的float tensor的index之间的映射
  std::unordered_map<int, int> quant_conversion_map_;
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

class Tensor : public GPUObject {  // CL中对算子存储的描述
  cl_mem memory_;
  cl_mem image_buffer_memory_;     // for TensorStorageType::IMAGE_BUFFER only
  bool memory_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
};

struct OperationDef {              // 定义一个CL层面op的src和des的tensor
  CalculationsPrecision precision; // 的dtype，storage_type，layout
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;
};

class GPUOperation {             // 一个CL层面Operation的定义，CL的各个Kernel里
  OperationDef definition_;      // 都继承了这个类，又实现了一些自己的东西
  std::vector<Tensor*> src_;
  std::vector<Tensor*> dst_;
  Arguments args_;
  std::vector<ElementwiseOperation*> linked_operations_;
};

struct GPUOperationWithRefs {              // 对GPUOperation的一份封装
  std::unique_ptr<GPUOperation> operation;
  std::vector<int> input_ids;
  std::vector<int> output_ids;
};

struct GPUOperationsSubgraph {                  // CL层面的子图
  std::vector<GPUOperationWithRefs> operations; // 这个子图中的GPUOperations
  std::vector<std::pair<BHWC, TensorDescriptor>> new_tensors; // 这个子图中的tensors
};                      
```

程序中是调用`InitializeOpenClApi()`函数进行OpenCL的初始化，其中首先调用`NewInferenceEnvironment()`创建一个默认的`InferenceEnvironment`，其实主要就是加载OpenCL Lib，获取设备Info，Context，配置CommandQueue，获取这个`cl_environment_`以及其的`properties`。

当`cl_enviroment_`确定之后，程序的下一个任务就是构建一个`InferenceBuilder`，代码中是根据delegation的options，graph来调用`cl::InferenceEnvironment::NewInferenceBuilder()`函数完成。该函数中创建了一个`InferenceBuilderImpl`类型的指针，并调用其`Initialize()`函数，用上面的options，graph，environment来初始化。具体来说，`Initialize()`函数中将这些options转为一个`CreateInferenceInfo`类型的变量`create_info`中，然后调用`InitFromGraph()`函数来设置`InferenceBuilder`中`InferenceContext`类型的对象`context_`，其核心代码如下：

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
  // 为更新后的`CLNode`分配`cl_mem`，其中一些对齐啊什么的处理全在里面
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