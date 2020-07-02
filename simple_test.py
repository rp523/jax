#coding: utf-8
import time
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.experimental.stax import serial, parallel, Dense, Sigmoid, FanOut, FanInSum, Identity
from jax.experimental import optimizers

from model.maker.model_maker import net_maker

def SkipDense(unit_num):
    return serial(FanOut(2), parallel(Dense(unit_num), Identity), FanInSum)

def Net():
    net = net_maker()
    activate = Sigmoid
    net.add_layer(serial(Dense(100), activate))
    for _ in range(1):
        net.add_layer(serial(Dense(100), activate))
    net.add_layer(Dense(1), name = "out")
    return net.get_jax_model()

def tgt_fun(x):
    return 1 * jnp.cos(x)

def main():
    LR = 1E-6
    BATCH_SIZE = 32
    EPOCH_NUM = 100000
    half = 4 * jnp.pi
    band = half * 2

    init_fun, apply_fun = Net()
    opt_init, opt_update, get_params = optimizers.sgd(LR)
    
    rng = jax.random.PRNGKey(0)
    _, init_params = init_fun(rng, (BATCH_SIZE, 1))
    opt_state = opt_init(init_params)

    def loss(params, x, y):
        p = apply_fun(params, x)["out"]
        assert(p.shape == y.shape)
        return jnp.abs((p - y)).sum() / x.shape[0]

    def update(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_val, grad_val = jax.value_and_grad(loss)(params, x, y)
        return loss_val, opt_update(i, grad_val, opt_state)

    t0 = time.time()
    for e in range(EPOCH_NUM):
        rng1, rng = jax.random.split(rng)
        x = jax.random.uniform(rng1, (BATCH_SIZE, 1)) * band - half
        y = tgt_fun(x)
        loss_val, opt_state = update(e, opt_state, x, y)
        t1 = time.time()
        if t1 - t0 > 1:
            t0 = t1
            print(e, loss_val)

    point_num = 100
    x = jnp.linspace(-half, half, point_num).reshape((-1, 1))
    y = tgt_fun(x)
    p = apply_fun(get_params(opt_state), x)["out"]
    plt.clf()
    plt.plot(x, y)
    plt.plot(x, p.flatten())
    plt.savefig("simple.png")

if __name__ == "__main__":
    main()
    print("Done.")