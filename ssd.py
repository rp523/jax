#coding: utf-8
import numpy as np
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

def Conv2WithSkip(  channel_size,
                    kernel_size,
                    stride,
                    ):
    ch = channel_size
    k = kernel_size
    s = stride
    Main = stax.serial( Conv(ch, (k, k), (s, s), "SAME"), BatchNorm(), Relu,
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
    return stax.serial( Conv2WithSkip(ch1, k, 1),
                        Conv2WithSkip(ch1, k, 1),
                        Conv(ch2, (k, k), (2, 2), "SAME"), Relu,)

def RootResNet18():
    net = net_maker()
    net.add_layer(Conv(64, (7, 7), (2, 2), "SAME"), name = "f2")    # stride = 2
    net.add_layer(StrideBlock( 64, 128, 3), name =  "f4")             # stride = 4
    net.add_layer(StrideBlock(128, 256, 3), name =  "f8")            # stride = 8
    net.add_layer(StrideBlock(256, 512, 3), name = "f16")           # stride = 16
    net.add_layer(StrideBlock(512, 512, 3), name = "f32")           # stride = 32
    return net

def SSD(pos_cls_vec, siz_vec, asp_vec):
    net = net_maker(prev_model = RootResNet18())
    anchor_num = siz_vec.size * asp_vec.size
    anchor_out = 4 + (1 + pos_cls_vec.size) # position of rect + classify
    out_ch = anchor_num * anchor_out
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f2", name =  "a2")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f4", name =  "a4")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f8", name =  "a8")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name = "f16", name = "a16")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name = "f32", name = "a32")
    return net
    
def make_batch_getter(batch_gen, batch_size, img_h, img_w):
    while True:
        images, _ = next(batch_gen)
        labels = np.random.uniform(size =  (batch_size, img_h//32, img_w//32, 512)) # dummy label
        yield images, labels

def main():
    rng_key = random.PRNGKey(0)
    
    BATCH_SIZE = 8
    IMG_H = 128
    IMG_W = 256
    INPUT_SHAPE = (BATCH_SIZE, IMG_H, IMG_W, 3)
    NUM_STEPS = 30
    MODEL_DIR = "model"
    NUM_CLASSES = 3
    
    init_fun, predict_fun = RootResNet18()
    _, init_params = init_fun(rng_key, INPUT_SHAPE)
    opt_init, opt_update, get_params = optimizers.adam(1E-3)
    
    rng = jax.random.PRNGKey(0)
    cityscapes = CityScapes(r"/mnt/hdd/dataset/cityscapes", rng, IMG_H, IMG_W)
    batch_gen = cityscapes.make_generator(  "train",
                                            label_txt_list = ["car", "person"],
                                            batch_size = BATCH_SIZE)
    batch_getter = make_batch_getter(batch_gen, BATCH_SIZE, IMG_H, IMG_W)

    def loss(params, batch):
        x, y = batch
        preds = predict_fun(params, x)
        y_ = preds["f32"]
        return jnp.sum((y - y_) ** 2)

    def _update(i, opt_state, batch):
        params = get_params(opt_state)
        loss_val, grad_val = value_and_grad(loss)(params, batch)
        return loss_val, opt_update(i, grad_val, opt_state)
    update = jit(_update)

    opt_state = opt_init(init_params)
    t0 = time.time()
    for i in range(NUM_STEPS):
        batch = next(batch_getter)
        loss_val, opt_state = update(i, opt_state, batch)
        t = time.time()
        print(i, "{:.1f}ms".format(1000 * (t - t0)), loss_val)
        t0 = t
    trained_params = get_params(opt_state)  # list format

if "__main__" == __name__:
    main()
    print("Done.")