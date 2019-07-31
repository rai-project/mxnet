#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=$3
NUMPREDS=5
DUPLICATE_INPUT=$(($NUMPREDS * $BATCHSIZE))
OUTPUTFOLDER=output_cpu
DATABASE_NAME=carml_cpu

cd ..

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi

go build -tags=nolibjpeg

export TF_CUDNN_USE_AUTOTUNE=0
export CARML_TF_DISABLE_OPTIMIZATION=0
export CUDA_LAUNCH_BLOCKING=0

./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --database_address=$DATABASE_ADDRESS --publish --disable_autotune=true --batch_size=$BATCHSIZE \
  --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME

export CUDA_LAUNCH_BLOCKING=1

# run framework trace to get acurate layer latency
./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --disable_autotune=true \
  --trace_level=FRAMEWORK_TRACE --database_name=$DATABASE_NAME
