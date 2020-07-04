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
    net.add_layer(serial(Dense(100), Swish()))
    for _ in range(1):
        net.add_layer(serial(Dense(100), Swish()))
    net.add_layer(Dense(1), name = "out")
    return net.get_jax_model()

def tgt_fun(x):
    return jnp.cos(x)# + jnp.sin(x.T[1])

def main(is_training):
    LR = 1E-3
    BATCH_SIZE = 32
    SAVE_PATH = "simple.bin"
    broaden_rate = 5
    half = 4 * jnp.pi * broaden_rate
    band = half * 2

    init_fun, apply_fun = Net()
    opt_init, opt_update, get_params = optimizers.adam(LR)
    
    rng = jax.random.PRNGKey(0)
    if os.path.exists(SAVE_PATH):
        init_params = pickle.load(open(SAVE_PATH, "rb"))
        print("LOADED INIT PARAMS")
    else:
        _, init_params = init_fun(rng, (BATCH_SIZE, 1))
    opt_state = opt_init(init_params)

    def loss(params, x, y):
        p = apply_fun(params, x)["out"]
        assert(p.shape == y.shape)
        return ((p - y) ** 2).sum() / x.shape[0]

    def update(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = jax.value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(i, grad_val, opt_state)
    
    t0 = time.time()
    e = 0
    while is_training:
        rng1, rng = jax.random.split(rng)
        x = jax.random.uniform(rng1, (BATCH_SIZE, 1)) * band - half
        y = tgt_fun(x / broaden_rate)
        loss_val, opt_state = update(e, opt_state, x, y)
        t1 = time.time()
        if t1 - t0 > 1:
            t0 = t1
            print(e, loss_val)
            pickle.dump(get_params(opt_state), open(SAVE_PATH, "wb"))
        e += 1

    point_num = 100
    x = jnp.linspace(-half, half, point_num).reshape((-1, 1))
    y = tgt_fun(x / broaden_rate)
    p = apply_fun(get_params(opt_state), x)["out"]
    plt.clf()
    plt.plot(x, y)
    plt.plot(x, p.flatten())
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")