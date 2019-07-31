#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=$3
FRAMEWORK_NAME=MXNet
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

echo "Start to run model analysis"

./mxnet-agent evaluation model info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --sort_output --format=csv,table --plot_all --output="$OUTPUTFOLDER/$MODELNAME/model_info"

echo "Start to run layer analysis"

./mxnet-agent evaluation layer info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv,table --plot_all --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_info"

./mxnet-agent evaluation layer aggre_info --framework_name=$FRAMEWORK_NAME --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv,table --plot_all --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_aggre_info"
