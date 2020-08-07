# Tflite在端侧CPU/GPU上运行全过程（一）

本文以Tflite自带的benchmark程序为例，说明一个来自converter的`.tflite`文件使用Tflite框架在端侧CPU/GPU上运行起来的全过程。该程序入口函数在`tensorflow/lite/tools/benchmark/benchmark_main.cc`中。本文包含五小节，其中前四节属于运行之前的初始化阶段，第五节是真正的运行阶段，各小节内容如下：

第一节：介绍Tflite框架读取解析模型文件得到Inference Workload的过程

第二节：介绍Tflite如何通过GPU Delegate的设置将Workload迁移到GPU上的过程

第三节：CPU算子到GPU端kernel的流程

第四节：GPU算子生成的细节

第五节：运行阶段的框架如何完成真正的推理计算

本节为第一节。



## 1. 编译运行

这里用到的Tflite代码可以用如下命令配置编译：

```ba
(base) cqchu@cqchu:~/compiler/TfliteRead$ ./configure 
You have bazel 3.1.0 installed.
Please specify the location of python. [Default is /home/cqchu/local/miniconda3/bin/python3]: 
Found possible Python library paths: /home/cqchu/local/miniconda3/lib/python3.7/site-packages
Please input the desired Python library path to use.  Default is [/home/cqchu/local/miniconda3/lib/python3.7/site-packages]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: y
Searching for NDK and SDK installations.

Please specify the home path of the Android NDK to use. [Default is /home/cqchu/Android/Sdk/ndk-bundle]: /home/cqchu/local/android-ndk-r18b

Please specify the (min) Android NDK API level to use. [Available levels: ['16', '17', '18', '19', '21', '22', '23', '24', '26', '27', '28']] [Default is 21]: 23

Please specify the home path of the Android SDK to use. [Default is /home/cqchu/Android/Sdk]: /home/cqchu/local/sdk

Please specify the Android SDK API level to use. [Available levels: ['27', '30']] [Default is 30]: 

Please specify an Android build tools version to use. [Available versions: ['30.0.0']] [Default is 30.0.0]: 

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
        --config=ngraph         # Build with Intel nGraph support.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=noaws          # Disable AWS S3 filesystem support.
        --config=nogcp          # Disable GCP support.
        --config=nohdfs         # Disable HDFS support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished

(base) cqchu@cqchu:~/compiler/TfliteRead$ bazel build -c opt --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
(base) cqchu@cqchu:~/compiler/TfliteRead$ bazel build -c opt --config android_arm64 tensorflow/lite:libtensorflowlite.so
(base) cqchu@cqchu:~/compiler/TfliteRead$ bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so
```

编译完成后可以用adb将生成的程序，库传到手机上测试，具体可以参考[run.sh](https://github.com/cqchu/TfliteRead/blob/r2.3/run.sh)。



## 2. 解析传入参数及读取模型

由于这部分些代码并不属于关键内容，故而也没细看，此处只是简要介绍流程。进入`main()`函数程序首先调用`ParseFlags()`解析传入进来的命令行参数，比如输入模型文件的路径，是否采用GPU，是否使用OpenCL作为后端等，这些参数解析完成后会放在一个全局变量`params_`当中。而后，程序进入`Run()`函数调用`LoadModel()`中的`BuildFromFile()`根据传入的tflite模型路径，基于flatbuffer库读取文件，解析出模型，最终返回一个`unique_ptr<FlatBufferModel>`类型的变量`model_`。



 ## 3. Interpreter及Subgraph的初始化

在完成模型解析以及模型读取后，程序调用`InitInterpreter()`开始进行Interpreter的初始化，而初始化的第一个工作就是算子的注册。

### 3.1 算子注册

Tflite中`BuiltinOpResolver`类维护框架中注册的算子，其继承自`MutableOpResolver`类，有两个`unordered_map`类型的变量`builtins_/custom_ops_`，分别用于维护内建的算子和用户自定义的算子。程序中调用`GetOpResolver()->BuiltinOpResolver()`完成算子注册，这个函数中部分代码如下：

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

`AddBuiltin()/AddCustom()`这个函数就是在向`unordered_map`类型的变量`builtins_/custom_ops_`中添加这`opcode --> TfLiteRegistration`的映射。在完成注册后，`GetOpResolver()`将这个`OpResolver`的指针返回，以用于后续构造`Interpreter`。

### 3.2 Tflite对神经网络中元素的描述抽象

```C++
/*******************************************************/
typedef struct TfLiteTensor {            // tflite对tensor的描述
  TfLiteType type;                       // 标志dtype，可以为fp16，fp32，int8，int32等
  TfLitePtrUnion data;                   // 一个由fp16，fp32，fp64，int8，int16，int32
                                         // int64，bool等类型指针以及void*构成的Union
  TfLiteIntArray* dims;                  // 用于记录tensor维度的类似于vector的结构体
  TfLiteAllocationType allocation_type;  // 标志内存分配策略
  size_t bytes;	                         // 存储这个Tensor所需要的字节数
  const char* name;                      // 这个tensor的名字
  struct TfLiteDelegate* delegate;       // 标记这个Tensor将在什么delegate
  TfLiteQuantization quantization; 	     // 量化相关参数
  TfLiteSparsity* sparsity;              // 稀疏性相关参数
} TfLiteTensor;

/*******************************************************/
typedef struct TfLiteNode {
  TfLiteIntArray* inputs;           // 这个Node的input的node的index
  TfLiteIntArray* outputs;          // 这个Node的output的node的index
  void* user_data;                  // 指向一个OpData的结构体，其是op的init()函数的返回值
                                    // 不同的算子维护的信息不同
  void* builtin_data;               //
  struct TfLiteDelegate* delegate;  // Node对应的delegate
} TfLiteNode;

/*******************************************************/
typedef struct TfLiteContext {
  size_t tensors_size;           // Context中tensor数量，其实就是subgraph以及interpreter
  								 // 的tensors vector的size()的值
  TfLiteTensor* tensors;         // 指向subgraph/interpreter的tensors_vector.data()
  int recommended_num_threads;   // 用于gemmlowp或者eigen的线程数设置
  bool allow_fp32_relax_to_fp16; // 允许用fp16代替fp32的标志位
} TfLiteContext;

/*******************************************************/
class Subgraph {
 private:
  std::vector<TfLiteTensor> tensors_; // subgraph中所有的Tensor构成的vector
  TfLiteContext context_ = {};        // subgraph维护的那个context
    
  // subgraph中Node到具体op的映射
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>> nodes_and_registration_;
  
  std::vector<int> inputs_;	    // input tensor在tensors_的下标
  std::vector<int> outputs_;    // output index在tensors_的下标
  std::vector<int> variables_;  // 是variable的tensor在tensors_的下标
  // pipeline执行算子时下一个要prepare node的下标
  int next_execution_plan_index_to_prepare_;
  // pipeline执行算子时下一个要分配内存node的下标
  int next_execution_plan_index_to_plan_allocation_; 
  // subgraph中所有node的一个拓扑排序
  std::vector<int> execution_plan_;   
  // 在apply delegate之前用于备份，以用于设置delegate失败后restore
  std::vector<int> pre_delegation_execution_plan_; 
  // 这个subgraph上成功创建的delegate
  std::vector<TfLiteDelegate*> delegates_applied_;  
  // 这个subgraphs_指针指向Interpreter::subgraphs_成员
  std::vector<std::unique_ptr<Subgraph>>* subgraphs_ = nullptr;
};

/*******************************************************/
class Interpreter {
 private:
  TfLiteContext* context_ = nullptr;                // interpreter维护的context_
  std::vector<TfLiteDelegatePtr> owned_delegates_;  // 配置好的Delegate的tensor
  std::vector<std::unique_ptr<Subgraph>> subgraphs_;// 这个interpreter所有的subgraphs
                                                    // 一般只有1个
};

```

上面列举了几个主要的数据结构，其中只列出了部分比较关键的成员，可以在看下面程序流程时返回参考。

### 3.3 Interpreter的初始化

转回正题，经过前面几步，我们已经获得了`FlatBufferModel`类型的模型变量`model_`，包含所有Op的`OpResolver`类型变量`resolver`，以及包含命令行参数的`params_`，现在程序将用这三个变量，对默认初始化的`Interpreter`变量`interpreter_`进行设置，这里是重载了`InterpreterBuilder`类的函数调用运算符。

```C++
tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, params_->num_threads); 
```

在这个函数里，其首先对传入的`params_`和`model_`做了一系列的检查，然后做了如下的事情，具体参考注释：

```C++
TfLiteStatus InterpreterBuilder::operator()(std::unique_ptr<Interpreter>* interpreter, int num_threads) {
  // 构建一个`model_`中各个op的`index --> TfLiteRegistration`的映射
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
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index]; // 获取`model_`的子图
    tflite::Subgraph* modified_subgraph = (*interpreter)->subgraph(subgraph_index); // 创建`interpreter_的子图`，然后用`model_`的子图来设置`interpreter_`的子图
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