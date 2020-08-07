####################### Compile #######################
bazel build -c opt --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
bazel build -c opt --config android_arm64 tensorflow/lite:libtensorflowlite.so
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so
##################### Compile End #####################

####################### Transfer ######################
src_path="/home/cqchu/compiler/TfliteRead/bazel-bin/tensorflow/lite"
litelib="${src_path}/libtensorflowlite.so"
benchmark="${src_path}/tools/benchmark/benchmark_model"
dst_path="/data/local/tmp/ccq/tflite"
model_path="/home/cqchu/models"

adb push ${litelib} ${dst_path}/lib/libtensorflowlite.so
adb push ${benchmark} ${dst_path}
adb push ${model_path}/*.tflite ${dst_path}/models
#################### Transfer End #####################

echo "######################################################################################################"
echo "############################################## RUN CODE ##############################################"
echo "######################################################################################################"

#################### Run Mobilenet ####################
# quant_model=mobilenet_v2_1.0_224_quant.tflite
# float_model=mobilenet_v2_1.0_224.tflite

# run_cmd="./benchmark_model --graph=models/${quant_model} \
#                            --num_threads=4 \
#                            --num_runs=10"
#                            --use_gpu=true \
#                            --gpu_precision_loss_allowed=true \
#                            --gpu_experimental_enable_quant=true \
#                            --gpu_backend=\"cl\""
# echo "cd ${dst_path} && ${run_cmd}" | adb shell

# run_cmd="./benchmark_model --graph=models/${float_model} \
#                            --num_threads=4 \
#                            --num_runs=10 \
#                            --use_gpu=true \
#                            --gpu_precision_loss_allowed=true \
#                            --gpu_experimental_enable_quant=true \
#                            --gpu_backend=\"cl\""
# echo "cd ${dst_path} && ${run_cmd}" | adb shell
#################### Mobilenet End ####################

################## Run OneNode Graph ##################
model=conv_fp32.tflite
run_cmd="./benchmark_model --graph=models/${model} \
                           --num_threads=4 \
                           --num_runs=1 \
                           --use_gpu=true \
                           --gpu_precision_loss_allowed=false \
                           --gpu_experimental_enable_quant=true \
                           --gpu_backend=\"cl\""
echo "cd ${dst_path} && ${run_cmd}" | adb shell
################## OneNode Graph End ##################
