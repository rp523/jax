#coding: utf-8
import os, time, pickle, argparse
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.experimental.stax import serial, parallel, Dense, Sigmoid, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental import optimizers

from model.maker.model_maker import net_maker

def SkipDense(unit_num):
    return serial(FanOut(2), parallel(Dense(unit_num), Identity), FanInSum)

def Swish():
    def init_fun(rng, input_shape):
        beta_shape = tuple(input_shape[1:])
        beta = jax.nn.initializers.ones(rng, beta_shape)
        params = (beta,)
        output_shape = input_shape
        return output_shape, params
    def apply_fun(params, inputs, **kwargs):
        beta = params[0]
        return inputs / (1.0 + jnp.exp(- beta * inputs))
    return init_fun, apply_fun

def ProdGaussian(scale):
    def init_fun(rng, input_shape_tuple, mu_init = jax.nn.initializers.zeros, sigma_init = jax.nn.initializers.ones):
        input_shape, x_shape = input_shape_tuple
        output_shape = input_shape
        k_mu, k_sigma = jax.random.split(rng)
        mu_shape = x_shape[1:]
        if not isinstance(mu_shape, tuple):
            mu_shape = (mu_shape,)
        mu = mu_init(k_mu, mu_shape)
        sigma = sigma_init(k_sigma, (1,)) * scale
        return output_shape, (mu, sigma)
    def apply_fun(params, input_tuple, **kwargs):
        mu, sigma = params
        base_val, x = input_tuple
        assert(x.ndim <= 2)
        if x.ndim > 1:
            axis = tuple(jnp.arange(1, x.ndim))
            assert(base_val.shape[1] == 1)
        else:
            axis = None
        power = -((x - mu) ** 2).sum(axis = axis).reshape(-1, 1) / (2.0 * (sigma ** 2))
        output_val = base_val * jnp.exp(power)
        return output_val
    return init_fun, apply_fun

def Net():
    unit_num = 1000
    net = net_maker()
    net.add_layer(serial(Dense(unit_num), Swish()))
    for _ in range(10):
        net.add_layer(serial(SkipDense(unit_num), Swish()))
    net.add_layer(serial(Dense(1)), name = "raw")
    net.add_layer(ProdGaussian(5), name = "out", input_name = ("raw", None))
    return net.get_jax_model()

def tgt_fun(x):
    r = (x.T[0] ** 2 + x.T[1] ** 2) ** 0.5
    y = jnp.zeros((x.shape[0], 1))
    y += jnp.exp(-(r - 5) ** 2 / 2).reshape((-1, 1))
    if x.ndim == 2:
        #y *= jnp.sin(x.T[1])
        pass
    return y * 1E-3

def main(is_training):
    LR = 1E-6
    BATCH_SIZE = 32
    X_DIM = 2
    SAVE_PATH = "simple.bin"
    broaden_rate = 1
    half = 4 * jnp.pi * broaden_rate
    band = half * 2

    init_fun, apply_fun = Net()
    opt_init, opt_update, get_params = optimizers.adam(LR)
    
    rng = jax.random.PRNGKey(0)
    if os.path.exists(SAVE_PATH):
        init_params = pickle.load(open(SAVE_PATH, "rb"))
        print("LOADED INIT PARAMS")
    else:
        _, init_params = init_fun(rng, (BATCH_SIZE, X_DIM))
    opt_state = opt_init(init_params)

    def loss(params, x, y):
        p = apply_fun(params, x)["out"]
        y = y.reshape((-1, 1))
        assert(p.shape == y.shape)
        return ((p - y) ** 2).sum() / x.shape[0]

    @jax.jit
    def update(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = jax.value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(i, grad_val, opt_state)
    
    def save_img(params):
        bin_num = 100
        plot_band = half * 1
        x = jnp.linspace(-plot_band, plot_band, bin_num)
        x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
        y = jnp.linspace(-plot_band, plot_band, bin_num)
        y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
        data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
        assert(data.shape == (bin_num * bin_num, 2))
        minus_E = apply_fun(params, data / broaden_rate)["out"]
        #minus_E = tgt_fun(data / broaden_rate)
        unnorm_log_q = minus_E
        unnorm_log_q = unnorm_log_q.reshape((bin_num, bin_num))
        X = jnp.linspace(-plot_band, plot_band, bin_num)
        Y = jnp.linspace(-plot_band, plot_band, bin_num)
        X, Y = jnp.meshgrid(X, Y)
        plt.clf()
        plt.pcolor(X, Y, unnorm_log_q)
        plt.colorbar()
        plt.savefig("simple.png")

    t0 = time.time()
    e = 0
    while is_training:
        rng1, rng = jax.random.split(rng)
        x = jax.random.uniform(rng1, (BATCH_SIZE, X_DIM)) * band - half
        y = tgt_fun(x / broaden_rate)
        loss_val, opt_state = update(e, opt_state, x, y)
        t1 = time.time()
        if t1 - t0 > 1:
            t0 = t1
            print(e, loss_val)
            pickle.dump(get_params(opt_state), open(SAVE_PATH, "wb"))
            save_img(get_params(opt_state))
        e += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")