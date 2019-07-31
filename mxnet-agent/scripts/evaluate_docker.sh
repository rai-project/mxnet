#!/bin/bash

DATABASE_ADDRESS=$1
DATABASE_NAME=resnet50_1_0
NUM_FILE_PARTS=10
MODEL_NAME=ResNet50
MODEL_VERSION=1.0
FRAMEWORK_NAME=caffe
TRACE_LEVEL=MODEL_TRACE
BATCH_SIZE=32
GPU=1

nvidia-docker run -t -v $HOME:/root carml/$FRAMEWORK_NAME-agent:amd64-gpu-latest predict dataset \
  --fail_on_error=true \
  --verbose \
  --publish=true \
  --publish_predictions=false \
  --gpu=$GPU \
  --num_file_parts=$NUM_FILE_PARTS \
  --batch_size=$BATCH_SIZE \
  --model_name=$MODEL_NAME \
  --model_version=$MODEL_VERSION \
  --database_address=$DATABASE_ADDRESS \
  --database_name=test \
  --trace_level=$TRACE_LEVEL
exit
