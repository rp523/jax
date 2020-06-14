#coding: utf-8
import numpy as np
import os, time

import jax
import jax.numpy as jnp
#from jax.config import config
from jax import grad, value_and_grad, device_put, tree_map
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

def SSD(pos_classes, siz_vec, asp_vec):
    net = net_maker(prev_model = RootResNet18())
    anchor_num = siz_vec.size * asp_vec.size
    anchor_out = 4 + (1 + len(pos_classes)) # position of rect + classify
    out_ch = anchor_num * anchor_out
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f2", name =  "a2")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f4", name =  "a4")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name =  "f8", name =  "a8")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name = "f16", name = "a16")
    net.add_layer(  Conv(out_ch, (3, 3), (1, 1), padding = "SAME"), input_name = "f32", name = "a32")
    return net

def main():
    BATCH_SIZE = 2
    SEED = 0
    LOOP_NUM = 100

    batch_size = BATCH_SIZE
    rng = jax.random.PRNGKey(SEED)
    ANCHOR_SIZ_NUM = 3
    siz_vec = 2 ** (np.arange(ANCHOR_SIZ_NUM) / ANCHOR_SIZ_NUM)

    ANCHOR_ASP_MAX = 2.0
    ANCHOR_ASP_NUM = 3
    asp_vec = ANCHOR_ASP_MAX ** np.linspace(-1, 1, ANCHOR_ASP_NUM)
    pos_classes = ["car", "person"]
    img_h = 128
    img_w = 256
    batch_size = batch_size
    init_fun, apply_fun = SSD(pos_classes, siz_vec, asp_vec).get_jax_model()

    rng1, rng = jax.random.split(rng)
    _, init_params = init_fun(rng1, (batch_size, img_h, img_w, 3))
    opt_init, opt_update, get_params = optimizers.adam(1E-4)
    
    rng1, rng = jax.random.split(rng)
    batch_getter = make_batch_getter(rng1, pos_classes, batch_size, siz_vec, asp_vec, img_h, img_w)

    opt_state = opt_init(init_params)

    def loss(params, x, y):
        preds = apply_fun(params, x)
        POS_ALPHA = 1.0
        def smooth_l1(x):
            return (0.5 * x ** 2) * (jnp.abs(x) < 1) + (jnp.abs(x) - 0.5) * (jnp.abs(x) >= 1)
        out = 0.0
        for stride in [2,4,8,16,32]:
            key = "a{}".format(stride)
            pred = preds[key]
            b, h, w, ch = pred.shape
            pred = pred.reshape(b, h, w, siz_vec.size * asp_vec.size, -1)
            pred_pos, pred_cls_logit = jnp.split(pred, [4], axis = -1)
            pos, pos_valid, cls, cls_valid = y[key]
            pos_diff = pred_pos - pos

            b, h, w, a = pos_valid.shape
            pos_valid = pos_valid.reshape(b, h, w, a, 1)
            out += (POS_ALPHA * smooth_l1(pos_diff) * pos_valid).sum()

            pred_cls = jax.nn.softmax(pred_cls_logit)
            b, h, w, a = cls_valid.shape
            cls_valid = cls_valid.reshape(b, h, w, a, 1)
            out += (- cls * jnp.log(pred_cls + 1E-10) * cls_valid).sum()
        return out

    @jax.jit
    def update(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(i, grad_val, opt_state)

    t0 = time.time()
    for i in range(LOOP_NUM):
        x, y = next(batch_getter)
        loss_val, opt_state = update(i, opt_state, x, y)
        t = time.time()
        print(i, "{:.1f}ms".format(1000 * (t - t0)), loss_val)
        t0 = t
    trained_params = get_params(opt_state)  # list format
    return trained_params
    
def arrange_annot(batched_labels, pos_classes, siz_vec, asp_vec, feat_h, feat_w):
    POS_IOU_TH = 0.5
    NEG_IOU_TH = 0.4
    batch_size = len(batched_labels)
    all_iou = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.float32)
    out_yc  = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.float32)
    out_xc  = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.float32)
    out_h   = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.float32)
    out_w   = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.float32)
    out_cls = np.zeros((batch_size, siz_vec.size, asp_vec.size, feat_h, feat_w), dtype = np.int32)

    base_yc = (np.arange(feat_h) + 0.5) / feat_h
    base_yc = np.tile(base_yc.reshape(-1, 1), (1, feat_w))
    base_xc = (np.arange(feat_w) + 0.5) / feat_w
    base_xc = np.tile(base_xc.reshape(1, -1), (feat_h, 1))

    for b, label in enumerate(batched_labels):
        for label_name, rects in label.items():
            for rect in rects:
                yc = rect[0]
                xc = rect[1]
                h  = rect[2]
                w  = rect[3]

                y0 = yc - h / 2
                y1 = yc + h / 2
                x0 = xc - w / 2
                x1 = xc + w / 2

                for s, siz in enumerate(siz_vec):
                    for a, asp in enumerate(asp_vec):
                        base_h = (1.0 / feat_h) * siz * (asp ** (-0.5))
                        base_w = (1.0 / feat_w) * siz * (asp ** ( 0.5))

                        base_y0 = base_yc - base_h / 2
                        base_y1 = base_yc + base_h / 2
                        base_x0 = base_xc - base_w / 2
                        base_x1 = base_xc + base_w / 2

                        iou = calc_iou(  base_y0, base_y1, base_x0, base_x1, base_h, base_w,
                                                y0, y1, x0, x1, h, w)
                        
                        update_feat_idx = (iou > all_iou[b, s, a])
                        if update_feat_idx.any():
                            out_yc[ b, s, a, update_feat_idx] = ((yc - base_yc) / base_h)[update_feat_idx]
                            out_xc[ b, s, a, update_feat_idx] = ((xc - base_xc) / base_w)[update_feat_idx]
                            out_h[  b, s, a, update_feat_idx] = np.log(h / base_h)
                            out_w[  b, s, a, update_feat_idx] = np.log(w / base_w)
                            out_cls[b, s, a, update_feat_idx] = 1 + pos_classes.index(label_name)
                            all_iou[b, s, a, update_feat_idx] = iou[update_feat_idx]

    out_cls[all_iou < NEG_IOU_TH] = 0
    # reshape
    all_iou = all_iou.transpose(0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size)
    out_yc  = out_yc.transpose( 0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size, 1)
    out_xc  = out_xc.transpose( 0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size, 1)
    out_h   = out_h.transpose(  0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size, 1)
    out_w   = out_w.transpose(  0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size, 1)
    out_cls = out_cls.transpose(0, 3, 4, 1, 2).reshape(batch_size, feat_h, feat_w, siz_vec.size * asp_vec.size)

    out_pos = np.append(np.append(out_yc, out_xc, axis = -1),
                        np.append(out_h , out_w , axis = -1),
                        axis = -1)
    out_cls = np.eye(1 + len(pos_classes))[out_cls]
    pos_valid = (POS_IOU_TH <= all_iou)
    cls_valid = np.logical_or(all_iou < NEG_IOU_TH, POS_IOU_TH <= all_iou)

    return out_pos, pos_valid, out_cls, cls_valid

def calc_iou(base_y0, base_y1, base_x0, base_x1, base_h, base_w,
                y0, y1, x0, x1, h, w):
    feat_h, feat_w = base_y0.shape
    overlap = np.logical_and(   np.logical_and(base_y0 <= y1, y0 <= base_y1),
                                np.logical_and(base_x0 <= x1, x0 <= base_x1))
    and_y0 = np.maximum(base_y0, np.ones((feat_h, feat_w)) * y0)
    and_y1 = np.minimum(base_y1, np.ones((feat_h, feat_w)) * y1)
    and_x0 = np.maximum(base_x0, np.ones((feat_h, feat_w)) * x0)
    and_x1 = np.minimum(base_x1, np.ones((feat_h, feat_w)) * x1)
    and_h  = and_y1 - and_y0
    and_w  = and_x1 - and_x0
    and_s  = and_h  * and_w

    base_s = base_h * base_w
    s      = h * w
    or_s   = base_s + s - and_s
    
    iou = np.zeros((feat_h, feat_w), dtype = np.float32)
    iou[overlap] = (and_s / or_s)[overlap]
    
    return iou

def make_batch_getter(rng, pos_classes, batch_size, siz_vec, asp_vec, img_h, img_w):
    dataset = CityScapes(r"/mnt/hdd/dataset/cityscapes", rng, img_h, img_w)
    batch_gen = dataset.make_generator( "train",
                                        label_txt_list = pos_classes,
                                        batch_size = batch_size)
    while True:
        images, batched_labels = next(batch_gen)
        labels = {}
        for stride in [2, 4, 8, 16, 32]:
            labels["a{}".format(stride)] = arrange_annot(batched_labels, pos_classes, siz_vec, asp_vec, img_h // stride, img_w // stride)
        yield images, labels

if "__main__" == __name__:
    main()
    print("Done.")