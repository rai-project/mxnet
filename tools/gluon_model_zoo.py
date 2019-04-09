#!env python

from gluoncv import model_zoo, data, utils
import os
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
import warnings
logging.basicConfig(level=logging.ERROR)


try:
    os.makedirs("/tmp/gluoncv")
except:
    pass


def export0(name, net, input):
    tmpdir = "/tmp/models/{}/".format(name)
    try:
        net.hybridize()
        net.forward(input)
        try:
            os.makedirs(tmpdir)
        except:
            pass
        net.export(tmpdir + "model")
        onnx_mxnet.export_model(
            tmpdir + "model-symbol.json",
            tmpdir + "model-0000.params",
            [input.shape],
            np.float32,
            "/tmp/onnx/{}_float32.onnx".format(name),
            # verbose=True,
        )
        print("wrote /tmp/onnx/{}_float32.onnx to disk".format(name))
    except Exception as inst:
        print(inst)
        # try:
        #     os.rmdir(tmpdir)
        # except:
        #     pass
    pass


def export(model_list, ctx, x, pretrained=True, **kwargs):
    pretrained_models = model_zoo.pretrained_model_list()
    for model in model_list:
        if model in pretrained_models:
            net = model_zoo.get_model(model, pretrained=True, **kwargs)
        else:
            net = model_zoo.get_model(model, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.initialize()
        net.collect_params().reset_ctx(ctx)
        net(x)
        mx.nd.waitall()
        export0(model, net, x)


def export_imagenet_models():
    ctx = mx.context.current_context()

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    models = [
        "resnet18_v1",
        "resnet34_v1",
        "resnet50_v1",
        "resnet101_v1",
        "resnet152_v1",
        "resnet18_v1b",
        "resnet34_v1b",
        "resnet50_v1b",
        "resnet50_v1b_gn",
        "resnet50_v1s",
        "resnet101_v1s",
        "resnet152_v1s",
        "resnet101_v1b",
        "resnet152_v1b",
        "resnet50_v1c",
        "resnet101_v1c",
        "resnet152_v1c",
        "resnet50_v1d",
        "resnet101_v1d",
        "resnet152_v1d",
        "resnet50_v1e",
        "resnet101_v1e",
        "resnet152_v1e",
        "resnet18_v2",
        "resnet34_v2",
        "resnet50_v2",
        "resnet101_v2",
        "resnet152_v2",
        "resnext50_32x4d",
        "resnext101_32x4d",
        "resnext101_64x4d",
    ]
    # Pruned ResNet
    models.extend([
        "resnet50_v1d_1.8x",
        "resnet50_v1d_3.6x",
        "resnet50_v1d_5.9x",
        "resnet50_v1d_8.8x",
        "resnet101_v1d_1.9x",
        "resnet101_v1d_2.2x",
    ])
    models.extend([
        "se_resnext50_32x4d",
        "se_resnext101_32x4d",
        "se_resnext101_64x4d",
        "se_resnet18_v1",
        "se_resnet34_v1",
        "se_resnet50_v1",
        "se_resnet101_v1",
        "se_resnet152_v1",
        "se_resnet18_v2",
        "se_resnet34_v2",
        "se_resnet50_v2",
        "se_resnet101_v2",
        "se_resnet152_v2",
        "senet_154",
        "squeezenet1.0",
        "squeezenet1.1",
        "mobilenet1.0",
        "mobilenet0.75",
        "mobilenet0.5",
        "mobilenet0.25",
        "mobilenetv2_1.0",
        "mobilenetv2_0.75",
        "mobilenetv2_0.5",
        "mobilenetv2_0.25",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "darknet53",
        "alexnet",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg16_atrous",
        "vgg19",
        "vgg19_bn",
    ])
    pretrained_models = model_zoo.pretrained_model_list()
    for m in pretrained_models:
        if not m in models:
            print(m)
    export(models, ctx, x)

    # 299x299
    x = mx.random.uniform(shape=(2, 3, 299, 299), ctx=ctx)
    models = ["inceptionv3", "nasnet_5_1538", "nasnet_7_1920", "nasnet_6_4032"]
    export(models, ctx, x)

    # 331x331
    x = mx.random.uniform(shape=(2, 3, 331, 331), ctx=ctx)
    models = ["nasnet_5_1538", "nasnet_7_1920", "nasnet_6_4032"]
    export(models, ctx, x)


def export_classification_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(2, 3, 32, 32), ctx=ctx)
    cifar_models = [
        "cifar_resnet20_v1",
        "cifar_resnet56_v1",
        "cifar_resnet110_v1",
        "cifar_resnet20_v2",
        "cifar_resnet56_v2",
        "cifar_resnet110_v2",
        "cifar_wideresnet16_10",
        "cifar_wideresnet28_10",
        "cifar_wideresnet40_8",
        "cifar_resnext29_32x4d",
        "cifar_resnext29_16x64d",
    ]
    export(cifar_models, ctx, x)


def export_ssd_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(
        shape=(1, 3, 512, 544), ctx=ctx
    )  # allow non-squre and larger inputs
    models = [
        "ssd_300_vgg16_atrous_voc",
        "ssd_512_vgg16_atrous_voc",
        "ssd_512_resnet50_v1_voc",
        "ssd_512_resnet101_v2_voc",
        "ssd_512_mobilenet1.0_voc",
        "ssd_300_vgg16_atrous_coco",
        "ssd_512_vgg16_atrous_coco",
        "ssd_512_resnet50_v1_coco",
        "ssd_512_mobilenet1.0_coco",
    ]
    export(models, ctx, x)


def export_faster_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(
        shape=(1, 3, 300, 400), ctx=ctx
    )  # allow non-squre and larger inputs
    models = [
        "faster_rcnn_resnet50_v1b_voc",
        "faster_rcnn_resnet50_v1b_coco",
        "faster_rcnn_resnet101_v1d_coco",
        "faster_rcnn_fpn_resnet50_v1b_coco",
        "faster_rcnn_fpn_bn_resnet50_v1b_coco",
        "faster_rcnn_fpn_resnet101_v1d_coco",
    ]
    export_model_list(models, ctx, x)


def export_mask_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 300, 400), ctx=ctx)
    models = [
        "mask_rcnn_resnet50_v1b_coco",
        "mask_rcnn_resnet101_v1d_coco",
        "mask_rcnn_fpn_resnet50_v1b_coco",
        "mask_rcnn_fpn_resnet101_v1d_coco",
    ]
    export(models, ctx, x)


def export_yolo3_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(
        shape=(1, 3, 320, 320), ctx=ctx
    )  # allow non-squre and larger inputs
    models = [
        "yolo3_darknet53_voc",
        "yolo3_mobilenet1.0_voc",
        "yolo3_mobilenet1.0_coco",
        "yolo3_darknet53_coco",
    ]
    export(models, ctx, x)


def export_segmentation_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    models = [
        "fcn_resnet101_coco",
        "fcn_resnet101_voc",
        "fcn_resnet50_ade",
        "fcn_resnet101_ade",
        "deeplab_resnet101_coco",
        "deeplab_resnet101_voc",
        "deeplab_resnet152_coco",
        "deeplab_resnet152_voc",
        "deeplab_resnet50_ade",
        "deeplab_resnet101_ade",
        "psp_resnet101_coco",
        "psp_resnet101_voc",
        "psp_resnet50_ade",
        "psp_resnet101_ade",
        "psp_resnet101_citys",
    ]
    export(models, ctx, x, pretrained=True, pretrained_base=True)


def export_simple_pose_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 256, 192), ctx=ctx)
    models = [
        "simple_pose_resnet18_v1b",
        "simple_pose_resnet50_v1b",
        "simple_pose_resnet101_v1b",
        "simple_pose_resnet152_v1b",
        "simple_pose_resnet50_v1d",
        "simple_pose_resnet101_v1d",
        "simple_pose_resnet152_v1d",
    ]
    export(models, ctx, x, pretrained=True, pretrained_base=True)


if __name__ == "__main__":
    pretrained_models = model_zoo.pretrained_model_list()
    print(pretrained_models)
    # export_imagenet_models()
    # export_segmentation_models()
    # export_yolo3_models()
    # export_mask_rcnn_models()
    # export_faster_rcnn_models()
    # export_ssd_models()
    # export_classification_models()
    # export_simple_pose_models()
