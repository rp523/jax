# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A mock-up showing a ResNet50 network with training on synthetic data.

This file uses the stax neural network definition library and the optimizers
optimization library.
"""

import numpy.random as npr

import jax.numpy as np
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)


# ResNet blocks compose other layers

def ConvBlock(kernel_size, filters, strides=(2, 2)):
  ks = kernel_size
  filters1, filters2, filters3 = filters
  Main = stax.serial(
      Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
      Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
      Conv(filters3, (1, 1)), BatchNorm())
  Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
  return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks

def ResNet50(num_classes):
  return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'), out_chan = 64, filter_shape = (7, 7), strides = (2, 2), padding = 'SAME'),
      BatchNorm(), Relu, MaxPool((3, 3), strides=(2, 2)),
      ConvBlock(3, [64, 64, 256], strides=(1, 1)),
      IdentityBlock(3, [64, 64]),
      IdentityBlock(3, [64, 64]),
      ConvBlock(3, [128, 128, 512]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      ConvBlock(3, [256, 256, 1024]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      ConvBlock(3, [512, 512, 2048]),
      IdentityBlock(3, [512, 512]),
      IdentityBlock(3, [512, 512]),
      AvgPool((7, 7)), Flatten, Dense(num_classes), LogSoftmax)


if __name__ == "__main__":
  rng_key = random.PRNGKey(0)

  BATCH_SIZE = 8
  NUM_CLASSES = 1001
  INPUT_SHAPE = (224, 224, 3, BATCH_SIZE)
  NUM_STEPS = 1000

  init_fun, predict_fun = ResNet50(NUM_CLASSES)
  _, init_params = init_fun(rng_key, INPUT_SHAPE)

  def loss(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    return -np.sum(logits * targets)

  def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(predict_fun(params, inputs), axis=-1)
    return np.mean(predicted_class == target_class)

  def synth_batches(batch_size):
    rng = npr.RandomState(0)
    while True:
      images = rng.rand(*INPUT_SHAPE).astype('float32')
      labels = rng.randint(NUM_CLASSES, size=(batch_size, 1))
      onehot_labels = labels == np.arange(NUM_CLASSES)
      yield images, onehot_labels

  opt_init, opt_update, get_params = optimizers.momentum(0.1, mass=0.9)
  batch_getter = synth_batches(BATCH_SIZE)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  opt_state = opt_init(init_params)
  for i in range(NUM_STEPS):
    print(i)
    opt_state = update(i, opt_state, next(batch_getter))
  trained_params = get_params(opt_state)

