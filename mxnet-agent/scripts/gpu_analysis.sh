#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODEL_NAME=$3
FRAMEWORK_NAME=MXNet
OUTPUT_FOLDER=output_gpu
DATABASE_NAME=carml_mxnet

cd ..

if [ ! -d $OUTPUT_FOLDER ]; then
  mkdir $OUTPUT_FOLDER
fi

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi
go build -tags=nolibjpeg

echo "Start to run model analysis"

./mxnet-agent evaluation model info --framework_name=$FRAMEWORK_NAME--database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --sort_output --format=csv,table --plot_all --output="$OUTPUT_FOLDER/$MODEL_NAME/model_info"

echo "Start to run layer analysis"

./mxnet-agent evaluation layer info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/layer_info"

./mxnet-agent evaluation layer aggre_info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/layer_aggre_info"

echo "Start to run gpu analysis"

./mxnet-agent evaluation gpu_kernel info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/gpu_kernel_info"

./mxnet-agent evaluation gpu_kernel name_aggre_info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/gpu_kernel_name_aggre_info"

./mxnet-agent evaluation gpu_kernel model_aggre_info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/gpu_kernel_model_aggre_info"

./mxnet-agent evaluation gpu_kernel layer_aggre_info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODEL_NAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output="$OUTPUT_FOLDER/$MODEL_NAME/$BATCHSIZE/gpu_kernel_layer_aggre_info"
