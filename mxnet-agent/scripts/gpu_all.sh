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
  MobileNet_0.5
  MobileNet_0.25
  MobileNet_0.75
  MobileNet_1.0
  MobileNet_0.5
  MobileNet_v2_0.25
  MobileNet_v2_0.5
  MobileNet_v2_0.75
  MobileNet_v2_1.0
  Inception_v3
  VGG16
  VGG19
  Faster_RCNN_ResNet50_v1b_VOC
  SSD_512_ResNet50_v1_COCO
  SSD_512_MobileNet_1.0_COCO
  SSD_512_VGG16_Atrous_COCO
  SSD_300_VGG16_Atrous_COCO
  SSD_512_ResNet50_v1_VOC
  DarkNet53
  DenseNet161
  ResNet18_v1
  ResNet18_v2
  ResNet34_v1
  ResNet34_v2
  ResNext50_32x4d
  ResNext101_32x4d
  SqueezeNet_v1.0
  SqueezeNet_v1.1
  Xception
  CIFAR_WideResNet16_10
  CIFAR_WideResNet28_10
  CIFAR_WideResNet40_8
)

for i in "${array[@]}"; do
  echo $i
  ./gpu_eval_ab.sh localhost 256 $i
  ./gpu_eval_fb.sh localhost 1 $i
  ./gpu_analysis.sh localhost 1 $i
done
