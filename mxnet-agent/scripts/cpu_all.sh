#!/usr/bin/env bash

declare -a array=(
  AlexNet
  DenseNet121
  ResNet50_v1
  ResNet101_v1
  ResNet152_v1
  ResNet50_v2
  ResNet101_v2
  ResNet152_v2
  MobileNet_1.0
  Inception_v3
  Faster_RCNN_ResNet50_v1b_VOC
  SSD_512_ResNet50_v1_COCO
  SSD_512_MobileNet_1.0_COCO
  SSD_512_VGG16_Atrous_COCO
  SSD_300_VGG16_Atrous_COCO
  VGG16 VGG19
)


for i in "${array[@]}"; do
  echo $i
  ./cpu_eval_ab.sh localhost 16 $i
  ./cpu_eval_fb.sh localhost 1 $i
done

for i in "${array[@]}"; do
  echo $i
  ./cpu_analysis.sh localhost 1 $i
done

