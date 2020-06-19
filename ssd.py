#coding: utf-8
import numpy as np
import os, time
from PIL import Image, ImageDraw

import jax
import jax.numpy as jnp
#from jax.config import config
from jax import grad, value_and_grad, device_put, tree_map
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                    FanOut, Flatten, GeneralConv, Identity,
                                    MaxPool, Relu, LogSoftmax, Softmax, elementwise)
from model.maker.model_maker import net_maker
from dataset.cityscapes import CityScapes
from checkpoint import CheckPoint
COLOR_LIST = [  (255,0,0),
                (0,255,0),
                (0,0,255),
                ]

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

def rescale(x):
    return (x / 255) * 2.0 - 1.0
Rescale = elementwise(rescale)

def RootResNet18():
    net = net_maker()
    net.add_layer(Rescale)
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
    BATCH_SIZE = 12
    SEED = 0
    EPOCH_NUM = 500

    rng = jax.random.PRNGKey(SEED)
    ANCHOR_SIZ_NUM = 3
    siz_vec = 2 ** (np.arange(ANCHOR_SIZ_NUM) / ANCHOR_SIZ_NUM)

    ANCHOR_ASP_MAX = 2.0
    ANCHOR_ASP_NUM = 3
    asp_vec = ANCHOR_ASP_MAX ** np.linspace(-1, 1, ANCHOR_ASP_NUM)

    pos_classes = ["car", "person"]
    all_class_num = 1 + len(pos_classes)
    img_h = 128
    img_w = 256
    batch_size = BATCH_SIZE
    init_fun, apply_fun = SSD(pos_classes, siz_vec, asp_vec).get_jax_model()

    rng1, rng = jax.random.split(rng)
    _, init_params = init_fun(rng1, (batch_size, img_h, img_w, 3))
    opt_init, opt_update, get_params = optimizers.adam(1E-5)
    
    rng1, rng = jax.random.split(rng)
    dataset = CityScapes(r"/mnt/hdd/dataset/cityscapes", rng, img_h, img_w)
    train_batch_getter = make_batch_getter(dataset, "train", rng1, pos_classes, batch_size, siz_vec, asp_vec, img_h, img_w)

    # yのクラスはSoftmaxによる正規化済
    def loss(params, x, y):
        # predsのクラスはまだ正規化されていないロジット値
        preds = apply_fun(params, x)
        POS_ALPHA = 1.0
        FOCAL_GAMMA = 2.0
        def smooth_l1(x):
            return (0.5 * x ** 2) * (jnp.abs(x) < 1) + (jnp.abs(x) - 0.5) * (jnp.abs(x) >= 1)
        out = 0.0
        for stride in [2,4,8,16,32]:
            key = "a{}".format(stride)
            pred = preds[key]
            b, h, w, _ = pred.shape
            pred = pred.reshape((b, h, w, siz_vec.size, asp_vec.size, 4 + all_class_num))
            pred_pos, pred_cls_logit = jnp.split(pred, [4], axis = -1)
            pos, pos_valid, cls, cls_valid = y[key]

            b, h, w, s, a = pos_valid.shape
            pos_valid = pos_valid.reshape(b, h, w, s, a, 1)
            assert(pred_pos.shape == (b, h, w, s, a, 4))
            out += (POS_ALPHA * smooth_l1(pred_pos - pos) * pos_valid).sum()

            #assert(cls.min() >= 0.0)
            #assert(cls.max() <= 1.0)
            pred_cls = jax.nn.softmax(pred_cls_logit, axis = -1)
            assert(pred_cls.shape == (b, h, w, s, a, all_class_num))
            b, h, w, s, a = cls_valid.shape
            cls_valid = cls_valid.reshape(b, h, w, s, a, 1)
            out += (- cls * ((1.0 - pred_cls) ** FOCAL_GAMMA) * jnp.log(pred_cls + 1E-10) * cls_valid).sum()
        # batch average
        out /= x.shape[0]

        #weight decay
        out += 1E-4 * net_maker.weight_decay(params)

        return out

    @jax.jit
    def update(cnt, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(cnt, grad_val, opt_state)

    src_dir = os.path.join("../ssd_checkpoint", "epoch{}".format(0))
    if os.path.exists(src_dir):
        init_params = CheckPoint.load_params(init_params, src_dir)

    opt_state = opt_init(init_params)
    itrnum_in_epoch = dataset.itrnum_in_epoch("train", batch_size)
    cnt = 0
    fori_num = 16
    loss_val = 0.0
    def body_fun(idx, old_info):
        _, opt_state = old_info
        x, y = next(train_batch_getter)
        loss_val, opt_state = update(idx, opt_state, x, y)
        return (loss_val, opt_state)
    t0 = time.time()
    for e in range(EPOCH_NUM):
        for l in range(itrnum_in_epoch // fori_num):
            # fori_loopまではメモリオーバーでjit化できない
            loss_val, opt_state = jax.lax.fori_loop(cnt, cnt + fori_num, body_fun, (loss_val, opt_state))
            cnt += fori_num
            t = time.time()
            print(  "epoch=[{}/{}]".format(e + 1, EPOCH_NUM),
                    "iter=[{}/{}]".format(l * fori_num + 1, itrnum_in_epoch),
                    "{:.1f}ms".format(1000 * (t - t0)),
                    loss_val)
            t0 = t
        dst_dir = os.path.join("../ssd_checkpoint", "epoch{}".format(e + 1))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        CheckPoint.save_params(get_params(opt_state), dst_dir)
    trained_params = get_params(opt_state)  # list format

    PROB_TH = 0.9
    test_batch_getter = make_batch_getter(dataset, "test", rng1, pos_classes, 1, siz_vec, asp_vec, img_h, img_w)
    stride_vec = [2,4,8,16,32]
    for l in range(itrnum_in_epoch):
        x, y = next(test_batch_getter)
        preds = apply_fun(trained_params, x)
        rects = feat2rects(preds, stride_vec, pos_classes, siz_vec, asp_vec, PROB_TH)
        visualize(rects, x, pos_classes, "vis", str(l))

    return trained_params

def visualize(rects_list, image_list, pos_classes, dst_dir, name_key):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_dir = os.path.abspath(dst_dir)
    for b, (image, rects_dict) in enumerate(zip(image_list, rects_list)):
        pil = Image.fromarray(image.astype(jnp.uint8))
        img_w, img_h = pil.size
        dr = ImageDraw.Draw(pil)
        for c, pos_class in enumerate(pos_classes):
            color = COLOR_LIST[c]
            for rect in rects_dict[pos_class]:
                yc = rect[0] * (img_h - 1)
                xc = rect[1] * (img_w - 1)
                h  = rect[2] * (img_h - 1)
                w  = rect[3] * (img_w - 1)
                y0 = yc - h / 2
                y1 = yc + h / 2
                x0 = xc - w / 2
                x1 = xc + w / 2
                dr.rectangle((x0, y0, x1, y1), outline = color, width = 1)
        dst_path = os.path.join(dst_dir, name_key + "_{}.png".format(b))
        pil.save(dst_path)
        print(dst_path)

def feat2rects(feat_dict, stride_vec, pos_classes, siz_vec, asp_vec, prob_th):
    all_out = None

    anchor_num = siz_vec.size * asp_vec.size
    for stride in stride_vec:
        # 特定のスケール特徴量マップのバッチ
        stride_key = "a{}".format(stride)
        batched_feat = feat_dict[stride_key]
        if batched_feat.ndim == 4:
            # 推論結果
            all_class_num = 1 + len(pos_classes)
            b, h, w, ch = batched_feat.shape
            batched_feat = batched_feat.reshape((b, h, w, siz_vec.size, asp_vec.size, 4 + all_class_num))
            pos, cls = jnp.split(batched_feat, [4], axis = -1)
            cls = jax.nn.softmax(cls)
        else:
            # 正解ラベルをそのまま読み込む
            # classについてはsoftmaxで正規化済である想定
            pos, pos_valid, cls, cls_valid = batched_feat
        batch_size, feat_h, feat_w, siz, asp, _ = pos.shape
        all_class_num = 1 + len(pos_classes)
        '''
        vecsize_per_anchor = 4 + all_class_num
        assert(feat_ch == anchor_num * vecsize_per_anchor)
        '''
        # initialize output (only once)
        if all_out is None:
            all_out = []
            for b in range(batch_size):
                out_dict = {}
                for pos_class in pos_classes:
                    out_dict[pos_class] = []
                all_out.append(out_dict)

        assert(pos.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 4))
        assert(cls.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, all_class_num))
        positive_cls = cls[:,:,:,:,:,1:] # remove negative probability
        max_positive_prob = np.max(positive_cls, axis = -1)
        max_positive_idx  = np.argmax(positive_cls, axis = -1)

        base_yc = (np.arange(feat_h) + 0.5) / feat_h
        base_yc = np.tile(base_yc.reshape(-1, 1), (1, feat_w))
        base_xc = (np.arange(feat_w) + 0.5) / feat_w
        base_xc = np.tile(base_xc.reshape(1, -1), (feat_h, 1))
        assert(base_yc.shape == base_xc.shape)
        if (prob_th < max_positive_prob).any():
            for b in range(batch_size):
                for s, siz in enumerate(siz_vec):
                    for a, asp in enumerate(asp_vec):
                        base_h = (1.0 / feat_h) * siz * (asp ** (-0.5))
                        base_w = (1.0 / feat_w) * siz * (asp ** ( 0.5))
                        for y_idx in range(feat_h):
                            for x_idx in range(feat_w):
                                max_prob = max_positive_prob[b, y_idx, x_idx, s, a]
                                if max_prob > prob_th:
                                    f_yc, f_xc, f_h, f_w = pos[b, y_idx, x_idx, s, a]
                                    rect_h = base_h * np.exp(f_h)
                                    rect_w = base_w * np.exp(f_w)
                                    rect_yc = base_yc[y_idx, x_idx] + base_h * f_yc
                                    rect_xc = base_xc[y_idx, x_idx] + base_w * f_xc
                                    pos_class = pos_classes[max_positive_idx[b, y_idx, x_idx, s, a]]
                                    rect = [rect_yc, rect_xc, rect_h, rect_w]
                                    all_out[b][pos_class].append(rect)
    return all_out

# index処理があるので、jax.numpyではなくnumpyで
def rects2feat(batched_annots, pos_classes, siz_vec, asp_vec, feat_h, feat_w):
    POS_IOU_TH = 0.5
    NEG_IOU_TH = 0.4
    batch_size = len(batched_annots)
    all_iou = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.float32)
    out_yc  = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.float32)
    out_xc  = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.float32)
    out_h   = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.float32)
    out_w   = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.float32)
    out_cls = np.zeros((batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size), dtype = np.int32)

    base_yc = (np.arange(feat_h) + 0.5) / feat_h
    base_yc = np.tile(base_yc.reshape(-1, 1), (1, feat_w))
    base_xc = (np.arange(feat_w) + 0.5) / feat_w
    base_xc = np.tile(base_xc.reshape(1, -1), (feat_h, 1))
    assert(base_yc.shape == base_xc.shape)

    for b, annots in enumerate(batched_annots):
        for label_name, rects in annots.items():
            for rect in rects:
                yc = rect[0]
                xc = rect[1]
                h  = rect[2]
                w  = rect[3]

                y0 = yc - h / 2
                y1 = yc + h / 2
                x0 = xc - w / 2
                x1 = xc + w / 2
                assert(y0 >= 0.0)
                assert(y0 <= 1.0)
                assert(y1 >= 0.0)
                assert(y1 <= 1.0)
                assert(x0 >= 0.0)
                assert(x0 <= 1.0)
                assert(x1 >= 0.0)
                assert(x1 <= 1.0)

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
                        
                        update_feat_idx = (iou > all_iou[b, :, :, s, a])
                        if update_feat_idx.any():
                            out_yc[ b, update_feat_idx, s, a] = ((yc - base_yc) / base_h)[update_feat_idx]
                            out_xc[ b, update_feat_idx, s, a] = ((xc - base_xc) / base_w)[update_feat_idx]
                            out_h[  b, update_feat_idx, s, a] = np.log(h / base_h)
                            out_w[  b, update_feat_idx, s, a] = np.log(w / base_w)
                            out_cls[b, update_feat_idx, s, a] = 1 + pos_classes.index(label_name)
                            all_iou[b, update_feat_idx, s, a] = iou[update_feat_idx]

    out_cls[all_iou < NEG_IOU_TH] = 0
    # reshape
    assert(all_iou.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size))
    out_yc  = out_yc.reshape( (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 1))
    out_xc  = out_xc.reshape( (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 1))
    out_h   = out_h.reshape(  (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 1))
    out_w   = out_w.reshape(  (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 1))
    assert(out_cls.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size))

    out_pos = np.append(np.append(out_yc, out_xc, axis = -1),
                        np.append(out_h , out_w , axis = -1),
                        axis = -1)
    assert(out_pos.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, 4))

    all_class_num = 1 + len(pos_classes)
    out_cls = np.eye(all_class_num, dtype = np.float32)[out_cls]
    assert(out_cls.shape == (batch_size, feat_h, feat_w, siz_vec.size, asp_vec.size, all_class_num))
    pos_valid = (POS_IOU_TH <= all_iou) # only positive
    cls_valid = np.logical_or(all_iou < NEG_IOU_TH, POS_IOU_TH <= all_iou) # positive & negative

    return out_pos, pos_valid, out_cls, cls_valid

def calc_iou(base_y0, base_y1, base_x0, base_x1, base_h, base_w,
                y0, y1, x0, x1, h, w):
    feat_h, feat_w = base_y0.shape
    assert((base_y0 <= base_y1).all())
    assert((base_x0 <= base_x1).all())
    assert((base_h > 0.0).all())
    assert((base_w > 0.0).all())
    assert(y0 <= y1)
    assert(x0 <= x1)

    overlap = np.logical_and(   np.logical_and(base_y0 <= y1, y0 <= base_y1),
                                np.logical_and(base_x0 <= x1, x0 <= base_x1))
    assert(overlap.shape == (feat_h, feat_w))
    ones = np.ones((feat_h, feat_w))
    and_y0 = np.maximum(base_y0, ones * y0)
    and_y1 = np.minimum(base_y1, ones * y1)
    and_x0 = np.maximum(base_x0, ones * x0)
    and_x1 = np.minimum(base_x1, ones * x1)
    and_h  = and_y1 - and_y0
    and_w  = and_x1 - and_x0
    and_s  = and_h  * and_w
    assert((and_h[overlap] >= 0.0).all())
    assert((and_w[overlap] >= 0.0).all())
    assert((and_s[overlap] >= 0.0).all())

    base_s = base_h * base_w
    s      = h * w
    or_s   = base_s + s - and_s
    assert((or_s[overlap] >= 0.0).all())

    iou = np.zeros((feat_h, feat_w), dtype = np.float32)
    iou[overlap] = and_s[overlap] / or_s[overlap]
    assert(iou.shape == (feat_h, feat_w))
    
    return iou

def make_batch_getter(dataset, dataset_type, rng, pos_classes, batch_size, siz_vec, asp_vec, img_h, img_w):
    batch_gen = dataset.make_generator( dataset_type,
                                        label_txt_list = pos_classes,
                                        batch_size = batch_size,
                                        aug_flip = True,
                                        aug_noise = True,
                                        aug_crop_y0 = 0.25,
                                        aug_crop_y1 = 0.75,
                                        aug_crop_x0 = 0.25,
                                        aug_crop_x1 = 0.75,
    )

    while True:
        images, batched_labels = next(batch_gen)
        labels = {}
        for stride in [2, 4, 8, 16, 32]:
            labels["a{}".format(stride)] = rects2feat(batched_labels, pos_classes, siz_vec, asp_vec, img_h // stride, img_w // stride)
        yield images, labels

# 学習とは別に、データのエンコード/デコードが正しいか確認
def label_encdec_test():
    rng = jax.random.PRNGKey(10)
    ANCHOR_SIZ_NUM = 3
    siz_vec = 2 ** (np.arange(ANCHOR_SIZ_NUM) / ANCHOR_SIZ_NUM)
    ANCHOR_ASP_MAX = 2.0
    ANCHOR_ASP_NUM = 3
    asp_vec = ANCHOR_ASP_MAX ** np.linspace(-1, 1, ANCHOR_ASP_NUM)
    pos_classes = ["car", "person"]
    #all_class_num = 1 + len(pos_classes)
    img_h = 128
    img_w = 256
    batch_size = 4
    dataset = CityScapes(r"/mnt/hdd/dataset/cityscapes", rng, img_h, img_w)
    batch_gen = dataset.make_generator( "train",
                                        label_txt_list = pos_classes,
                                        batch_size = batch_size,
                                        aug_flip = True,
                                        aug_crop_y0 = 0.25,
                                        aug_crop_y1 = 0.77,
                                        aug_crop_x0 = 0.25,
                                        aug_crop_x1 = 0.75,
                                        )
    for i in range(100):
        images, batched_labels = next(batch_gen)
        feat_dict = {}
        stride_vec = [2, 4, 8, 16, 32]
        for stride in stride_vec:
            feat_dict["a{}".format(stride)] = rects2feat(batched_labels, pos_classes, siz_vec, asp_vec, img_h // stride, img_w // stride)
        batched_rects = feat2rects(feat_dict, stride_vec, pos_classes, siz_vec, asp_vec, 0.5)
        visualize(batched_rects, images, pos_classes, "eval", str(i))

if "__main__" == __name__:
    main()
    print("Done.")