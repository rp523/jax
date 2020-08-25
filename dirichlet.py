#coding: utf-8
import os, time, pickle
import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity
from jax.experimental.optimizers import adam
from jax.scipy.special import gammaln, digamma
import hydra
from dataset.mnist import Mnist
from dataset.fashion_mnist import FashionMnist
from model.maker.model_maker import net_maker

def Swish():
    def init_fun(rng, input_shape):
        beta_shape = tuple(input_shape[1:])
        beta = jax.nn.initializers.ones(rng, beta_shape)
        output_shape = input_shape
        return output_shape, (beta,) # output_shape, params
    def apply_fun(params, inputs, **kwargs):
        beta = params[0]
        x = inputs
        out = x * jax.nn.sigmoid(beta * x)
        assert(out.shape == x.shape)
        return out
    return init_fun, apply_fun
def SkipDense(unit_num):
    return serial(FanOut(2), parallel(Dense(unit_num), Identity), FanInSum)
def nn(class_num):
    return serial(  Flatten,
                    Dense(300), Swish(),
                    SkipDense(300), Swish(),
                    SkipDense(300), Swish(),
                    Dense(class_num)
                    )
    return serial(  Conv(16, (7, 7), (2, 2), 'VALID'), Tanh,
                    Conv(32, (3, 3), (2, 2), 'VALID'), Tanh,
                    Conv(64, (3, 3), (2, 2), 'VALID'), Tanh,
                    Flatten,
                    Dense(100), Tanh,
                    Dense(class_num)
                    )

def show_curve():
    rng = jax.random.PRNGKey(0)
    rng_m, rng_f = jax.random.split(rng)
    learned_mnist   = Mnist(       rng_m, batch_size = 1, data_type = "test", one_hot = False, dequantize = True, flatten = False, dir_path = ".", remove_classes = [0], remove_col_too = True)
    unlearned_mnist = Mnist(       rng_m, batch_size = 1, data_type = "test", one_hot = False, dequantize = True, flatten = False, dir_path = ".", remove_classes = np.arange(1,10), remove_col_too = True)
    fashion_mnist   = FashionMnist(rng_f, batch_size = 1, data_type = "test", one_hot = False, dequantize = True, flatten = False, dir_path = ".")
    assert(np.array(  learned_mnist.sample(get_all = True)[1] != 0).all())
    assert(np.array(unlearned_mnist.sample(get_all = True)[1] == 0).all())

    point_num = 100
    _, apply_fun = nn(9)
    classify_weight_path = "/home/isgsktyktt/work/outputs/2020-08-24/00-45-30/params.bin"
    classify_params = pickle.load(open(classify_weight_path, "rb"))

    # dataset loop
    for unlearned_data, unlearned_text in zip(  [unlearned_mnist, fashion_mnist],
                                                ["number0", "fasihon"]):
        # model loop
        for loss_type, last_layer, weight_path in zip(
                                                [
                                                    "dilichlet_cross_entropy",
                                                    "dilichlet_cross_entropy",
                                                    "dilichlet_L2",
                                                    "dilichlet_L2",
                                                ],
                                                [
                                                    "softmax",
                                                    "relu",
                                                    "softmax",
                                                    "relu",
                                                ],
                                                [
                                                    "/home/isgsktyktt/work/multirun/2020-08-24/01-08-41/4/params.bin",
                                                    "/home/isgsktyktt/work/multirun/2020-08-24/01-08-41/5/params.bin",
                                                    "/home/isgsktyktt/work/multirun/2020-08-24/01-08-41/14/params.bin",
                                                    "/home/isgsktyktt/work/multirun/2020-08-24/01-08-41/15/params.bin",
                                                ]
                                                ):
            params = pickle.load(open(weight_path, "rb"))

            learned_x,   learned_y   = learned_mnist.sample(get_all = True)
            unlearned_x, unlearned_y = unlearned_data.sample(get_all = True)

            learned_logit   = apply_fun(params, learned_x  )
            unlearned_logit = apply_fun(params, unlearned_x)
            classify_learned_logit   = apply_fun(classify_params, learned_x  )
            classify_unlearned_logit = apply_fun(classify_params, unlearned_x)

            classify_learned_softmax   = jax.nn.softmax(classify_learned_logit  )
            classify_unlearned_softmax = jax.nn.softmax(classify_unlearned_logit)

            if last_layer == "softmax":
                learned_alpha   = jnp.exp(  learned_logit)
                unlearned_alpha = jnp.exp(unlearned_logit)
                learned_softmax   = jax.nn.softmax(learned_logit)
                unlearned_softmax = jax.nn.softmax(unlearned_logit)
            elif last_layer == "relu":
                learned_alpha   = jax.nn.relu(learned_logit  ) + 1.0
                unlearned_alpha = jax.nn.relu(unlearned_logit) + 1.0
                learned_softmax    = learned_alpha   / learned_alpha.sum(  axis = -1, keepdims = True)
                unlearned_softmax  = unlearned_alpha / unlearned_alpha.sum(axis = -1, keepdims = True)
            else:
                assert(0)
            learned_certainty   = learned_alpha.sum(  axis = -1)
            unlearned_certainty = unlearned_alpha.sum(axis = -1)
            learned_argmax          = learned_logit.argmax(         axis = -1) + 1
            classify_learned_argmax = classify_learned_logit.argmax(axis = -1) + 1
            
            learned_certainty = np.array(learned_certainty)
            learned_argmax = np.array(learned_argmax)
            learned_y  = np.array(learned_y)
            unlearned_certainty = np.array(unlearned_certainty)
            unlearned_y  = np.array(unlearned_y)
            assert(  learned_y.shape == learned_argmax.shape)
            assert(  learned_y.shape == learned_certainty.shape)
            assert(unlearned_y.shape == unlearned_certainty.shape)

            plt.clf()

            # dummy loop for classify
            for model_type, l_logit_max, u_logit_max, l_softmax_max, u_softmax_max, l_certainty, u_certainty, l_argmax in [[
                                                    "",
                                                    classify_learned_logit.max(axis = -1),
                                                    classify_unlearned_logit.max(axis = -1),
                                                    classify_learned_softmax.max(axis = -1),
                                                    classify_unlearned_softmax.max(axis = -1),
                                                    None,
                                                    None,
                                                    classify_learned_argmax,
                                                ],
                                                [
                                                    "dilichlet_",
                                                    learned_logit.max(axis = -1),
                                                    unlearned_logit.max(axis = -1),
                                                    learned_softmax.max(axis = -1),
                                                    unlearned_softmax.max(axis = -1),
                                                    learned_certainty,
                                                    unlearned_certainty,
                                                    learned_argmax,
                                                ],
                                                ]:
                # output loop
                for metric_name, l_metric, u_metric in [
                                                    ["logit",   l_logit_max, u_logit_max],
                                                    ["softmax", l_softmax_max, u_softmax_max],
                                                    ["evidence", l_certainty, u_certainty],
                                                ]:
                    if (model_type == "") and (metric_name == "evidence"):
                        continue
                    metric_idx = (np.linspace(0.0, 1.0, point_num) * (l_metric.size + u_metric.size - 1)).astype(np.int)
                    metric_vec = (np.sort(np.append(l_metric, u_metric)))[metric_idx]
                    plot_x = np.zeros(metric_vec.shape)
                    plot_y = np.zeros(metric_vec.shape)
                    for m, metric_th in enumerate(metric_vec):
                        accuracy = 0.0
                        is_confident = (l_metric >= metric_th)
                        if is_confident.any():
                            accuracy = np.sum((learned_y == l_argmax)[is_confident]) / learned_y.size
                        reject_rate = 0.0
                        is_unlearned = (u_metric < metric_th)
                        if is_unlearned.any():
                            reject_rate = is_unlearned.mean()
                        plot_x[m] = reject_rate
                        plot_y[m] = accuracy
                    plt.plot(plot_x, plot_y, label = model_type + metric_name)
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            graph_text = last_layer + "_" + loss_type
            plt.title(graph_text)
            graph_name = graph_text + ".png"
            dst_dir_path = os.path.join("dst", unlearned_text)
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
            graph_path = os.path.join(dst_dir_path, graph_name)
            print(graph_path)
            plt.savefig(graph_path)

@hydra.main("dirichlet.yaml")
def main(cfg):
    print(cfg.pretty())
    seed = cfg.optim.seed
    batch_size = cfg.optim.batch_size
    lr = cfg.optim.lr
    focal_gamma = cfg.optim.focal_gamma
    epoch_num = cfg.optim.epoch_num
    weight_decay_rate = cfg.optim.weight_decay_rate
    burnin_epoch = cfg.optim.burnin_epoch
    kl_weight = cfg.optim.kl_weight
    log_sec = cfg.optim.log_sec
    weight_name = cfg.optim.weight_name
    loss_type = cfg.optim.loss_type
    remove_class = cfg.data.remove_class
    remove_col_too = cfg.data.remove_col_too

    init_fun, apply_fun = nn(10 - int(remove_col_too) * len(remove_class))
    rng = jax.random.PRNGKey(seed)

    rng_train, rng_test, rng_param, rng = jax.random.split(rng, 4)
    train = Mnist(rng_train, batch_size, "train", one_hot =  True, dequantize = True, flatten = False, dir_path = hydra.utils.get_original_cwd(), remove_classes = remove_class, remove_col_too = remove_col_too)
    test  = Mnist(rng_test,           1,  "test", one_hot = False, dequantize = True, flatten = False, dir_path = hydra.utils.get_original_cwd(), remove_classes = remove_class, remove_col_too = remove_col_too)
    opt_init, opt_update, get_params = adam(lr)
    input_shape = (batch_size, 28, 28, 1)
    _, init_params = init_fun(rng_param, input_shape)
    opt_state = opt_init(init_params)

    def accuracy(opt_state, x, y):
        params = get_params(opt_state)
        y_pred_idx = apply_fun(params, x).argmax(axis = -1) + int(remove_col_too)
        if remove_col_too:
            for rem_val in remove_class:
                add_idxs = jnp.where(y_pred_idx >= rem_val)[0]
                if add_idxs.any():
                    pred_idx = jax.ops.index_add(y_pred_idx, add_idxs, 1)
        return (y == y_pred_idx).mean()
    def ce_loss(params, x, y):
        y_pred = apply_fun(params, x)
        y_pred = jax.nn.softmax(y_pred)
        ce = (- y * ((1.0 - y_pred) ** focal_gamma) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        loss = ce
        loss += weight_decay_rate * net_maker.weight_decay(params)
        return loss
    def kl_to_ones(y, alpha):
        alpha_ = y + (1.0 - y) * alpha
        s_ = alpha_.sum(axis = -1, keepdims = True)
        t1 = gammaln(s_).sum(axis = -1)
        t2 = -(jnp.log(alpha_)).sum(axis = -1)
        t3 = ((alpha_ - 1.0) * (digamma(alpha_) - digamma(s_))).sum(axis = -1)
        kl = (t1 + t2 + t3).mean()
        return kl
    def calc_alpha(params, x):
        y_pred = apply_fun(params, x)
        if cfg.model.last == "softmax":
            # the same achitecture as conventional ones
            alpha = jnp.exp(y_pred)
        elif cfg.model.last == "relu":
            # paper-written
            evidence = jax.nn.relu(y_pred)
            alpha = evidence + 1.0
        return alpha
    def dirichlet_l2_loss(params, x, y, prio_weight):
        alpha = calc_alpha(params, x)
        s = alpha.sum(axis = -1, keepdims = True)
        p = alpha / s
        loss = ((y - p) ** 2 + p * (1.0 - p) / (s + 1.0)).sum(axis = -1).mean()
        loss += weight_decay_rate * net_maker.weight_decay(params)
        loss += prio_weight * kl_to_ones(y, alpha)
        return loss
    def dirichlet_ce_loss(params, x, y, prio_weight):
        alpha = calc_alpha(params, x)
        s = alpha.sum(axis = -1, keepdims = True)
        loss = y * (digamma(s) - digamma(alpha))
        loss = loss.sum(axis = -1).mean()
        loss += weight_decay_rate * net_maker.weight_decay(params)
        loss += prio_weight * kl_to_ones(y, alpha)
        return loss
    @jax.jit
    def update(idx, opt_state, x, y, prio_weight):
        params = get_params(opt_state)
        if loss_type == "dili_ce":
            loss_val, grad_val = jax.value_and_grad(dirichlet_ce_loss)(params, x, y, prio_weight)
        elif loss_type == "dili_l2":
            loss_val, grad_val = jax.value_and_grad(dirichlet_l2_loss)(params, x, y, prio_weight)
        else:
            loss_val, grad_val = jax.value_and_grad(ce_loss)(params, x, y)
        isnan_grad = net_maker.isnan_params(grad_val)
        isnan_loss = jnp.isnan(loss_val)
        isnan_flg  = jnp.logical_or(isnan_grad, isnan_loss)
        def valid_update_fun(state):
            grad_val, opt_state = state
            return opt_update(idx, grad_val, opt_state)
        def invalid_update_fun(state):
            grad_val, opt_state = state
            return opt_state
        def valid_incr_fun(idx):
            return idx + 1
        def invalid_incr_fun(idx):
            return idx
        opt_state = jax.lax.cond(isnan_flg, invalid_update_fun, valid_update_fun, (grad_val, opt_state))
        idx       = jax.lax.cond(isnan_flg, invalid_incr_fun,   valid_incr_fun,   idx)
        return idx, loss_val, opt_state, isnan_grad
    
    proc_epoch = 0.0
    idx = 0
    t0 = time.time()
    run_loss = 0.0
    run_cnt = 0
    val_acc_max = 0
    while True:
        prio_weight = min(kl_weight, proc_epoch / burnin_epoch)
        x, y = train.sample()
        idx, loss_val, opt_state, isnan_grad = update(idx, opt_state, x, y, prio_weight)
        if jnp.logical_not(isnan_grad):
            run_loss += loss_val
            run_cnt += 1
            proc_epoch += (batch_size / 60000)
            t1 = time.time()
            if ((t1 - t0 > log_sec) or (proc_epoch > epoch_num)) and (run_cnt > 0):
                x, y = test.sample(get_all = True)
                log_txt = ""
                acc = accuracy(opt_state, x, y)
                for t, txt in enumerate(["epoch={:.2f}".format(proc_epoch),
                                        "loop={}".format(idx),
                                        "loss={:.3f}".format(run_loss / run_cnt),
                                        "acc={:.2f}%".format(acc * 100),
                                        ]):
                    if t > 0:
                        log_txt += ","
                    log_txt += txt
                if proc_epoch > burnin_epoch:
                    if acc > val_acc_max:
                        val_acc_max = acc
                        pickle.dump(get_params(opt_state), open(weight_name, "wb"))
                with open("log.txt", "a") as f:
                    f.write("{}\n".format(log_txt))
                print(log_txt)
                run_loss = 0.0
                run_cnt = 0
                t0 = t1
            if proc_epoch > epoch_num:
                break

if __name__ == "__main__":
    #show_curve()
    main()
    print("Done.")
