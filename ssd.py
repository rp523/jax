#coding: utf-8
import numpy as onp
import os, time

import jax
import jax.numpy as jnp
#from jax.config import config
from jax import jit, grad, random, value_and_grad, device_put, tree_map
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax, Softmax)
from model.maker.model_maker import net_maker
from dataset.cityscapes import CityScapes

def Conv2WithSkip(channel_size,
                  kernel_size,
                  stride):
    ch = channel_size
    k = kernel_size
    s = stride
    Main = stax.serial(
                       Conv(ch, (k, k), (s, s), "SAME"), BatchNorm(), Relu,
                       Conv(ch, (k, k), (s, s), "SAME"), BatchNorm(), Relu,
                       )
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu,)

def StrideBlock(channel1,
                channel2,
                kernel_size,
                ):
    ch1 = channel1
    ch2 = channel2
    k = kernel_size
    return stax.serial(Conv2WithSkip(ch1, k, 1),
                       Conv2WithSkip(ch1, k, 1),
                       Conv(ch2, (k, k), (2, 2), "SAME"), Relu,)

def RootResNet18():
    net = net_maker()
    # stride = 1
    net.add_layer(Conv(64, (7, 7), (2, 2), "SAME"))
    # stride = 2
    net.add_layer(Identity, name = "f2")
    net.add_layer(StrideBlock(64, 128, 3))
    # stride = 4
    net.add_layer(Identity, name = "f4")
    net.add_layer(StrideBlock(128, 256, 3))
    # stride = 8
    net.add_layer(Identity, name = "f8")
    net.add_layer(StrideBlock(256, 512, 3))
    # stride = 16
    net.add_layer(Identity, name = "f16")
    net.add_layer(StrideBlock(512, 512, 3))
    # stride = 32
    net.add_layer(Identity, name = "f32", is_output = True)
    return net.get_jax_model()

def make_batch_getter(batch_gen, batch_size, img_h, img_w):
    while True:
        images, _ = next(batch_gen)
        labels = onp.random.uniform(size =  (batch_size, img_h//32, img_w//32, 512)) # dummy label
        yield images, labels

def main():
    rng_key = random.PRNGKey(0)
    
    BATCH_SIZE = 2
    IMG_H = 64
    IMG_W = 128
    INPUT_SHAPE = (BATCH_SIZE, IMG_H, IMG_W, 3)
    NUM_STEPS = 30
    MODEL_DIR = "model"
    NUM_CLASSES = 3
    
    init_fun, predict_fun = RootResNet18()
    _, init_params = init_fun(rng_key, INPUT_SHAPE)
    opt_init, opt_update, get_params = optimizers.adam(1E-3)
    
    rng = jax.random.PRNGKey(0)
    cityscapes = CityScapes(r"/mnt/hdd/dataset/cityscapes", rng, IMG_H, IMG_W)
    batch_gen = cityscapes.make_generator("train",
                                          label_txt_list = ["car", "person"],
                                          batch_size = BATCH_SIZE)
    batch_getter = make_batch_getter(batch_gen, BATCH_SIZE, IMG_H, IMG_W)

    def loss(params, batch):
        x, y = batch
        preds = predict_fun(params, x)
        y_ = preds["f32"]
        return jnp.sum((y - y_) ** 2)

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

if "__main__" == __name__:
    main()
    print("Done.")