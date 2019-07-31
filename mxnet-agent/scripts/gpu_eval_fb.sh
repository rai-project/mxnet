#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=$3
NUMPREDS=5
DUPLICATE_INPUT=$(($NUMPREDS * $BATCHSIZE))
OUTPUTFOLDER=output_gpu
DATABASE_NAME=carml
GPU_DEVICE_ID=0

cd ..

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi
go build -tags=nolibjpeg

export TF_CUDNN_USE_AUTOTUNE=0
export CARML_TF_DISABLE_OPTIMIZATION=0
export CUDA_LAUNCH_BLOCKING=0

./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true --batch_size=$BATCHSIZE \
  --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID

export CUDA_LAUNCH_BLOCKING=1

# run framework trace to get acurate layer latency
./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=FRAMEWORK_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID

export CUDA_LAUNCH_BLOCKING=0

# run system library trace to get acurate gpu kernel latency
./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID

# run system library trace with gpu metrics to get metrics of each cuda kernel
# https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference-7x

# Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=achieved_occupancy --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID

# Number of single-precision floating-point operations executed by non-predicated threads
# (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.
# The count does not include special operations.
./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=flop_count_sp --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID

./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=dram_read_bytes,dram_write_bytes --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID
