import argparse


import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import export_block

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval

parser = argparse.ArgumentParser(
    description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
opt = parser.parse_args()


# Load Model
model_name = opt.model
pretrained = True
net = get_model(model_name, pretrained=pretrained)

ctx = mx.context.current_context()
# 224x224
input = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)

tmpdir = "/tmp/models/{}/".format(model_name)
net.hybridize()
pred = net.forward(input)

net.export(tmpdir + "model")
