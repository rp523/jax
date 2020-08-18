#coding: utf-8
import os, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity
from jax.experimental.optimizers import adam
from jax.scipy.special import gammaln, digamma
import hydra
from dataset.mnist import Mnist
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
    log_sec = cfg.optim.log_sec
    weight_name = cfg.optim.weight_name
    loss_type = cfg.optim.loss_type

    init_fun, apply_fun = nn(10)
    rng = jax.random.PRNGKey(seed)

    rng_train, rng_test, rng_param, rng = jax.random.split(rng, 4)
    train = Mnist(rng_train, batch_size, "train", one_hot =  True, dequantize = True, flatten = False, dir_path = hydra.utils.get_original_cwd())
    test  = Mnist( rng_test,          1,  "test", one_hot = False, dequantize = True, flatten = False, dir_path = hydra.utils.get_original_cwd())
    opt_init, opt_update, get_params = adam(lr)
    input_shape = (batch_size, 28, 28, 1)
    _, init_params = init_fun(rng_param, input_shape)
    opt_state = opt_init(init_params)

    def accuracy(opt_state, x, y):
        params = get_params(opt_state)
        y_pred = apply_fun(params, x).argmax(axis = -1)
        return (y == y_pred).mean()
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
    def dirichlet_l2_loss(params, x, y, prio_weight):
        y_pred = apply_fun(params, x)
        alpha = jnp.exp(y_pred)
        s = alpha.sum(axis = -1, keepdims = True)
        p = alpha / s
        loss = ((y - p) ** 2 + p * (1.0 - p) / (s + 1.0)).sum(axis = -1).mean()
        loss += weight_decay_rate * net_maker.weight_decay(params)
        loss += prio_weight * kl_to_ones(y, alpha)
        return loss
    def dirichlet_ce_loss(params, x, y, prio_weight):
        y_pred = apply_fun(params, x)
        alpha = jnp.exp(y_pred)
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
        opt_state = opt_update(idx, grad_val, opt_state)
        return (idx + 1), loss_val, opt_state
    
    proc_epoch = 0.0
    idx = 0
    t0 = time.time()
    run_loss = 0.0
    run_cnt = 0
    while True:
        prio_weight = min(1.0, proc_epoch / burnin_epoch)
        x, y = train.sample()
        idx, loss_val, opt_state = update(idx, opt_state, x, y, prio_weight)
        run_loss += loss_val
        run_cnt += 1
        proc_epoch += (batch_size / 60000)
        t1 = time.time()
        if ((t1 - t0 > log_sec) or (proc_epoch > epoch_num)) and (run_cnt > 0):
            x, y = test.sample(get_all = True)
            log_txt = ""
            for t, txt in enumerate(["epoch={:.2f}".format(proc_epoch),
                                    "loop={}".format(idx),
                                    "loss={:.3f}".format(run_loss / run_cnt),
                                    "acc={:.2f}%".format(accuracy(opt_state, x, y) * 100),
                                    ]):
                if t > 0:
                    log_txt += ","
                log_txt += txt
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
    main()
    print("Done.")
