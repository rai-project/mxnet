#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=$3
NUMPREDS=5
OUTPUTFOLDER=output_cpu
DATABASE_NAME=carml_cpu

cd ..

if [ ! -d $OUTPUTFOLDER ]; then
  mkdir $OUTPUTFOLDER
fi

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi

go build -tags=nolibjpeg

export TF_CUDNN_USE_AUTOTUNE=0
export CARML_TF_DISABLE_OPTIMIZATION=0
export CUDA_LAUNCH_BLOCKING=0

# run model trace to get acurate model latency and throughput
for ((b = 1; b <= $BATCHSIZE; b *= 2)); do
  ./mxnet-agent predict urls --model_name=$MODELNAME --duplicate_input=$(($NUMPREDS * $b)) --database_address=$DATABASE_ADDRESS --publish --disable_autotune=true --batch_size=$b \
    --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME
done
