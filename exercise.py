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

import numpy as onp
import os, time

import jax
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random, value_and_grad, device_put, tree_map
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax, Softmax)

from dataset.cityscapes import CityScapes

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
    # Since output channel depends on where this block is called,
    # Do not fix it until block allocation is detected.
    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks

def ResNet50(num_classes):
    return stax.serial(
        Conv(64, (7, 7), (2, 2), 'SAME'),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides=(2, 2)),
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
        AvgPool((7, 7)),
        Flatten,
        Dense(num_classes),
        Softmax)

        
def main():
    rng_key = random.PRNGKey(0)
    
    BATCH_SIZE = 8
    NUM_CLASSES = 1001
    INPUT_SHAPE = (BATCH_SIZE, 224, 224, 3)
    NUM_STEPS = 30
    MODEL_DIR = "model"
    
    init_fun, predict_fun = ResNet50(NUM_CLASSES)
    _, init_params = init_fun(rng_key, INPUT_SHAPE)

    def loss(params, batch):
        inputs, targets = batch
        preds = predict_fun(params, inputs)
        return -jnp.sum(targets * jnp.log(preds + 1E-10))

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=-1)
        predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
        return jnp.mean(predicted_class == target_class)

    def make_batch_getter(batch_size):
        rng = onp.random.RandomState(0)
        cityscapes = CityScapes(r"/mnt/hdd/dataset/cityscapes", 256, 512)
        gen = cityscapes.make_generator("train",
                                        label_txt_list = ["car", "person"],
                                        batch_size = batch_size,
                                        seed = 0)
        while True:
            images, _ = next(gen)
            labels = rng.randint(NUM_CLASSES, size=(batch_size, 1))
            onehot_labels = labels == jnp.arange(NUM_CLASSES)
            yield images, onehot_labels

    opt_init, opt_update, get_params = optimizers.momentum(0.1, mass=0.9)
    batch_getter = make_batch_getter(BATCH_SIZE)

    # slightly faster than jax-jit
    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        loss_val, grad_val = value_and_grad(loss)(params, batch)
        return loss_val, opt_update(i, grad_val, opt_state)
    
    opt_state = opt_init(init_params)
    for i in range(NUM_STEPS):
        t0 = time.time()
        batch = next(batch_getter)
        loss_val, opt_state = update(i, opt_state, batch)
        t1 = time.time()
        print(i, "{:.1f}ms".format(1000 * (t1 - t0)), loss_val)
    trained_params = get_params(opt_state)  # list format

if __name__ == "__main__":
    main()
    print("Done.")

