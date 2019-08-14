#!/usr/bin/env bash

declare -a array=(
  # DenseNet121
  # ResNet50_v1
  # ResNet101_v1
  # ResNet152_v1
  # ResNet50_v2
  # ResNet101_v2
  # ResNet152_v2
  # MobileNet_0.25
  # MobileNet_0.5
  # MobileNet_0.75
  # MobileNet_1.0
  VGG16
  VGG19
  # Faster_RCNN_ResNet50_v1b_VOC
  # MobileNet_v2_0.25
  # MobileNet_v2_0.5
  # MobileNet_v2_0.75
  # MobileNet_v2_1.0
  # AlexNet
  # Inception_v3
  # SSD_512_ResNet50_v1_COCO
  # SSD_512_MobileNet_1.0_COCO
  # SSD_512_VGG16_Atrous_COCO
  # SSD_300_VGG16_Atrous_COCO
  # SSD_512_ResNet50_v1_VOC
  # DarkNet53
  # DenseNet161
  # ResNet18_v1
  # ResNet18_v2
  # ResNet34_v1
  # ResNet34_v2
  # ResNext50_32x4d
  # ResNext101_32x4d
  # SqueezeNet_v1.0
  # SqueezeNet_v1.1
  # CIFAR_WideResNet16_10
  # CIFAR_WideResNet28_10
  # CIFAR_WideResNet40_8
)

for i in "${array[@]}"; do
  echo $i
  # ./gpu_eval_ab.sh localhost 8 $i
  ./gpu_eval_fb.sh localhost 32 $i
  ./gpu_analysis.sh localhost 32 $i
done

# echo DenseNet121
# ./gpu_eval_fb.sh localhost 32 DenseNet121
# ./gpu_analysis.sh localhost 32 DenseNet121

# echo Faster_RCNN_ResNet50_v1b_VOC
# ./gpu_eval_fb.sh localhost 4 Faster_RCNN_ResNet50_v1b_VOC
# ./gpu_analysis.sh localhost 4 Faster_RCNN_ResNet50_v1b_VOC

# declare -a array2=(
#   ResNet50_v1
#   ResNet101_v1
#   ResNet152_v1
#   ResNet50_v2
#   ResNet101_v2
#   ResNet152_v2
#   VGG16
#   VGG19
# )

# for i in "${array2[@]}"; do
#   echo ResNet50_v1
#   ./gpu_eval_fb.sh localhost 256 $i
#   ./gpu_analysis.sh localhost 256 $i
# done

# declare -a array3=(
#   MobileNet_0.25
#   MobileNet_0.5
#   MobileNet_0.75
#   MobileNet_1.0
# )

# for i in "${array3[@]}"; do
#   echo ResNet50_v1
#   ./gpu_eval_fb.sh localhost 64 $i
#   ./gpu_analysis.sh localhost 64 $i
# done
