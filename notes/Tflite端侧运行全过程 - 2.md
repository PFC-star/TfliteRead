# Tflite在端侧CPU/GPU上运行全过程（二）

本文以Tflite自带的benchmark程序为例，说明一个来自converter的`.tflite`文件使用Tflite框架在端侧CPU/GPU上运行起来的全过程。该程序入口函数在`tensorflow/lite/tools/benchmark/benchmark_main.cc`中。本文包含五小节，其中前四节属于运行之前的初始化阶段，第五节是真正的运行阶段，各小节内容如下：

第一节：介绍Tflite框架读取解析模型文件得到Inference Workload的过程

第二节：介绍Tflite如何通过GPU Delegate的设置将Workload迁移到GPU上的过程

第三节：CPU算子到GPU端kernel的流程

第四节：GPU算子生成的细节

第五节：运行阶段的框架如何完成真正的推理计算

本节为第二节

## 4. GPU Delegate的设置

### 4.1 Delegate的创建

承接上文，到此框架已经完成了CPU上对Workload的设置，现在来看程序如何将这个Workload转给GPU。其从设置GPU代理开始，程序中会遍历所有的`delegate_provider`(COREML, Default-NoDelegate, EXTERNAL, GPU, Hexagon, NNAPI, XNNPACK七种provider以单例模式注册在系统中)，根据`param_`中维护的传入的delegate信息，初始化相应的delegate，此处就是GPU。

程序中调用`CreateTfLiteDelegate()`获取一个`TFLiteDelegate`类型指针，这个指针指向一个`GPUDelegate`。具体来说这个函数中首先调用`TfLiteGpuDelegateOptionsV2Default()`获取一个默认的GPU配置。然后根据传入的参数，修改这个默认GPU配置，比如是否允许精度损失，是否量化，cl还是gl等等。最后再调用`CreateGPUDelegate()->TfLiteGpuDelegateV2Create()`根据最终的GPU配置，创建GPU代理，即new了一个`tflite::gpu::Delegate()`。这个`tflite::gpu::Delegate`类有一个`struct TFLiteDelegate`类型的结构体，这个结构体里有个`data_`指针，这个指针就指向自己所属的这个`tflite::gpu::Delegate`对象，最后最终将这个`TFLiteDelegate`结构体的指针返回回去了。也不知道为啥要设计的这么绕。

### 4.2 更新Subgraph

在完成了Delegate的创建后，程序将根据Delegate更新原来的Subgrah，这一部分是将workload由CPU端转到GPU端的关键，入口函数是`interpreter_->ModifyGraphWithDelegate()`，其中遍历每个subgraph，调用每个subgraph的`ModifyGraphWithDelegate()`。

在这个函数中首先有个`RedoAllDelegates()`函数，重新re-apply所有的delegate，但如果之前程序并没有调用`UndoAllDelegate()`则不会执行，一般只有在配置Delegate失败的时候会出现这种情况，正常来说并不执行。

下一步程序调用`PrepareOpsStartingAt()`，这个函数里面以拓扑序遍历这个subgraph中的node，并调用这个node的`prepare()`函数，完成了各个算子的准备。这个准备其实就是对各个算子的输入输出做一些检查，并对算子的`OpData`结构体中一些配置进行了设置。

在之后调用`SwitchToDelegateContext()`，其实就是`subgraph->context_`几个函数指针的赋值，这几个函数指针在CPU版本本来是为空的，在Delegate的情况下却是需要的，所以此处做一个设置然后进行后续的设置。

经过简单设置后，最终程序调用`delegate->Prepare()`函数，这其实是个函数指针，其指向的真实的函数是`tflite::gpu::Delegate().DelegatePrepare()`，这个函数完成了真实的workload delegate的工作，具体如下。

1. 创建了一个`TfLiteRegistration`对象，定义几个lambda赋值给这个对象的`init`，`free`，`prepare`，`invoke`四个函数指针实现，此处列举其核心代码。

    ```C++
    const TfLiteRegistration kRegistration = {
        // .init func
        [](TfLiteContext* context, const char* buffer, size_t) -> void* {
            const auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
            auto* gpu_delegate = GetDelegate(params->delegate);
            const auto status = gpu_delegate->Prepare(context, params);     // 重要，后面会用到
        },
        // .free func
        [](TfLiteContext*, void* buffer) -> void {
            delete reinterpret_cast<DelegateKernel*>(buffer);
        },
        // .prepare func
        [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
            auto* gpu_delegate_kernel = GetDelegateKernel(node);
            const auto status = gpu_delegate_kernel->GetRequiredTemporaries(
                context, node, &node->temporaries);
        },
        // .invoke
        [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
            const auto status = GetDelegateKernel(node)->Invoke(context);    // 重要，后面会讲到
        },
      };
    ```

2. 获取一下这个`tflite::gpu::Delegate`的指针

3. `GetOpsToReplace()` - 这个函数查找`subgraph->nodes_and_registrations_`中哪些op需要被replace成GPU版本。

    * 为了实现这个功能，该函数里定义了`GraphPartitionHelper`对象，然后调用其的`Partition()`函数对根据subgraph进行划分，得到一系列包含若干个算子的op的子集`NodeSubset`，其数据结构如下

      ```C++
      struct NodeSubset {
        std::vector<int> nodes;           // 这个Subset中的node的index
        std::vector<int> input_tensors;   // 这个Subset中的input tensor的下标，一般来说是
                                          // 所有Node的input tensor unique后的结果
        std::vector<int> output_tensors;  // 这个Subset中的output tensor的下标
      };
      ```

    * 其划分的依据就是由若干个不支持的算子将原来的图分为几部分，具体来说，上面这个对象有两个变量，`supported_nodes_`和`unsupported_nodes_info`，分别表示在这个delegate上支持和不支持的那些算子信息。这些信息由`PrepareSupportedNodes()`函数遍历所有算子得到，其判断每个算子支持与否并记录在相关数据结构中。

    * 然后调用`PreviewDelegatePartitioning()->PartitionGraphIntoIndependentNodeSubsets()`对上面算子进行划分，因为测试用的mobilenet_v2所有算子GPU都支持，所以此处实际上将整张图变为一个Subset，这个subset中包含subgraph中所有的算子以及整图input和output的index。若有些不支持，其可能会根据不支持算子的位置，将网络划分为多个Subset，然后有些在GPU上执行，有些在CPU上执行

    * 这个函数同时还找到那些需要被replace的算子的index，并在最后会将之返回

    * 最后调用`PopulatePreviewDelegateParams()`将划分的结果(比如delegate)存在`subgraph->partitioning_preview_cache_`中

4. **`ReplaceNodeSubsetsWithDelegateKernels()`** 上面已经找到了知道了具体的delegate，需要替换的op的index，并且在上面创建了一个`TfLiteRegistration`对象，现在程序中利用这三个信息来替换CPU node为GPU Kernel，并对应的修改了`execution_plan_`，由于内容太多，请参考下面一小节。

5. 替换完成后这里会调用`SwitchToKernelContext()`，将`subgraph`的`context_`中部分函数指针设为delegation相关函数，然后释放一些不要的信息

6. 将设置好的这个delegate的指针存入`subgraph->delegates_applied_`这个vector中

到此，GPU上的设置就完成了。此时`execution_plan_`中只有一个Node，其op_code为`BuiltinOperator_DELEGATE`，而这个Node中又包含一个GPU Kernel List，之后程序推理时就是顺序执行这个GPU Kernel List。