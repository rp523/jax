#coding: utf-8
import os, time, pickle, argparse
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.experimental.stax import serial, parallel, Dense, Sigmoid, FanOut, FanInSum, Identity
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

def Net():
    net = net_maker()
    net.add_layer(serial(Dense(300), Swish()))
    for _ in range(1):
        net.add_layer(serial(Dense(300), Swish()))
    net.add_layer(Dense(1), name = "out")
    return net.get_jax_model()

def tgt_fun(x):
    print(x.shape)
    y = jnp.cos(x.T[0])#
    if x.ndim == 2:
        #y *= jnp.sin(x.T[1])
        pass
    return y

def main(is_training):
    LR = 1E-3
    BATCH_SIZE = 32
    X_DIM = 1
    SAVE_PATH = "simple.bin"
    broaden_rate = 4
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

    #@jax.jit
    def update(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = jax.value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(i, grad_val, opt_state)
    
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
        e += 1

    bin_num = 100
    plot_band = half * 1
    x = jnp.linspace(-plot_band, plot_band, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-plot_band, plot_band, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))
    #minus_E = apply_fun(init_params, data / broaden_rate)["out"]
    minus_E = tgt_fun(data / broaden_rate)
    unnorm_log_q = minus_E
    unnorm_log_q = unnorm_log_q.reshape((bin_num, bin_num))
    print(unnorm_log_q.min(), unnorm_log_q.max())
    X = jnp.linspace(-plot_band, plot_band, bin_num)
    Y = jnp.linspace(-plot_band, plot_band, bin_num)
    X, Y = jnp.meshgrid(X, Y)
    plt.pcolor(X, Y, unnorm_log_q)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")