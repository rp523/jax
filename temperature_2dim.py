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
                    Dense(128), Swish(),
                    Dense(128), Swish(),
                    Dense(class_num + 1)
                    )

def show_result(test, apply_fun, const_temp):
    weight_path = "/home/isgsktyktt/work/outputs/2020-09-14/18-45-44/params.bin"
    params = pickle.load(open(weight_path, "rb"))
    x_min = -10.0
    x_max =  10.0
    x_grid = 0.25
    x_num = int((x_max - x_min) / x_grid)
    y_min = -10.0
    y_max =  10.0
    y_grid = 0.25
    y_num = int((y_max - y_min) / y_grid)

    x = np.zeros((y_num, x_num, 2))
    for yi in range(y_num):
        for xi in range(x_num):
            x[yi, xi, 0] = y_min + yi * y_grid
            x[yi, xi, 1] = x_min + xi * x_grid
    x = x.reshape((-1, 2))
    applied = apply_fun(params, x)
    logit = applied[:,:-1]
    if const_temp is None:
        beta = jnp.exp(applied[:,-1]).reshape((logit.shape[0], 1))
    else:
        beta = const_temp
    y_all = jax.nn.softmax(beta * logit)
    y = y_all[:, 1].reshape((y_num, x_num))
    #y = beta.reshape((y_num, x_num))
    #y = (y - y.min()) / (y.max() - y.min())

    print(y.min(), y.max())
    plt.clf()
    for yi in range(y_num):
        for xi in range(x_num):
            s = y[yi, xi]
            assert(s >= 0.0)
            assert(s <= 1.0)
            r = min(1.0, (      s) * 2)
            b = min(1.0, (1.0 - s) * 2)
            g = min(r, b)
            r = np.clip(r, 0.0, 1.0)
            g = np.clip(g, 0.0, 1.0)
            b = np.clip(b, 0.0, 1.0)
            plt.fill_between(   [x_min + (xi + 0) * x_grid, x_min + (xi + 1) * x_grid],
                                [y_min + (yi + 0) * y_grid, y_min + (yi + 0) * y_grid],
                                [y_min + (yi + 1) * y_grid, y_min + (yi + 1) * y_grid],
                                facecolor = (r, g, b),
            )
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.show()


class TwoNormal:
    def __init__(self, param_list, rng):
        self.__param_list = param_list
        self.__rng = rng
    def sample(self):
        rng_c, rng_g, self.__rng = jax.random.split(self.__rng, 3)
        c = jax.random.randint(rng_c, (1, ), 0, len(self.__param_list))
        mu, sigma = self.__param_list[int(c)]
        val = jax.random.normal(rng_g, (2,))
        val = jnp.array(mu) + val * jnp.array(sigma)
        return c, val

class Dataset:
    def __init__(self, rng, batch_size, one_hot):
        self.__sample_num = 1000
        sigma = 1.0
        two_norm = TwoNormal([
                            [[0.0, -1.5 * sigma], [sigma, sigma]],
                            [[0.0,  1.5 * sigma], [sigma, sigma]],
                            ],
                            rng)
        self.__x = np.zeros((self.__sample_num, 2))
        self.__y = np.zeros((self.__sample_num, 2))
        for i in range(self.__sample_num):
            c, val = two_norm.sample()
            self.__x[i] = np.array(val)
            self.__y[i, c] = 1.0
        if one_hot is False:
            self.__y = self.__y.argmax(axis = 1)
        self.__batch_size = batch_size
        self.__rng = rng
    def sample(self, get_all = False):
        self.__rng, rng = jax.random.split(self.__rng)
        idx = jax.random.randint(rng, (self.__batch_size,), 0, self.__sample_num)
        if get_all is False:
            x = self.__x[idx]
            y = self.__y[idx]
            return x, y
        else:
            return self.__x, self.__y

@hydra.main("temperature_2dim.yaml")
def main(cfg):
    print(cfg.pretty())
    seed = cfg.optim.seed
    batch_size = cfg.optim.batch_size
    lr = cfg.optim.lr
    focal_gamma = cfg.optim.focal_gamma
    epoch_num = cfg.optim.epoch_num
    burnin_epoch = cfg.optim.burnin_epoch
    const_temp = cfg.optim.const_temp
    weight_decay_rate = cfg.optim.weight_decay_rate
    log_sec = cfg.optim.log_sec
    weight_name = cfg.optim.weight_name
    
    init_fun, apply_fun = nn(2)
    rng = jax.random.PRNGKey(seed)

    rng_train, rng_test, rng_param, rng = jax.random.split(rng, 4)
    train  = Dataset(rng_train, batch_size, one_hot =  True)
    test   = Dataset(rng_test,           1, one_hot = False)

    show_result(test, apply_fun, const_temp)
    exit()

    opt_init, opt_update, get_params = adam(lr)
    input_shape = (batch_size, 2)
    _, init_params = init_fun(rng_param, input_shape)
    opt_state = opt_init(init_params)

    def accuracy(opt_state, x, y):
        params = get_params(opt_state)
        y_pred_idx = apply_fun(params, x)[:,:-1].argmax(axis = -1)
        return (y == y_pred_idx).mean()
    def ce_loss(params, x, y):
        applied = apply_fun(params, x)
        logit = applied[:,:-1]
        if const_temp is None:
            beta = jnp.exp(applied[:,-1]).reshape((logit.shape[0], 1))
        else:
            beta = const_temp
        y_pred = jax.nn.softmax(beta * logit)
        ce = (- y * ((1.0 - y_pred) ** focal_gamma) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        loss = ce
        loss += weight_decay_rate * net_maker.weight_decay(params)
        return loss
    @jax.jit
    def update(idx, opt_state, x, y):
        params = get_params(opt_state)
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
        x, y = train.sample()
        idx, loss_val, opt_state, isnan_grad = update(idx, opt_state, x, y)
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
    main()
    print("Done.")
