#coding: utf-8
import time
import numpy as np
import jax
from jax import numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Tanh, elementwise
from ebm.sampler import Sampler
from model.maker.model_maker import net_maker

def swish(x):
    return x / (1.0 + jnp.exp(- x))
Swish = elementwise(swish)

def mlp(out_ch):
    net = net_maker()
    net.add_layer(Dense(300))
    net.add_layer(Swish)
    net.add_layer(Dense(300))
    net.add_layer(Swish)
    net.add_layer(Dense(out_ch), name = "out")
    return net.get_jax_model()

def main():
    LR = 1E-3
    LAMBDA = 0.5
    BATCH_SIZE = 10
    X_SIZE = 2
    rng = jax.random.PRNGKey(0)
    q_init_fun, q_apply_fun_raw = mlp(1)
    f_init_fun, f_apply_fun_raw = mlp(X_SIZE)
    def q_apply_fun(params, x):
        ret = q_apply_fun_raw(params, x)["out"]    # batch_ 1
        assert(ret.size == 1)
        return ret.sum()
    def f_apply_fun(params, x):
        return f_apply_fun_raw(params, x)["out"]
    q_init, q_update, q_get_params = optimizers.adam(LR)
    f_init, f_update, f_get_params = optimizers.adam(LR)

    rng_q, rng_f, rng = jax.random.split(rng, 3)
    _, q_init_params = q_init_fun(rng_q, input_shape = (BATCH_SIZE, X_SIZE))
    rng1, rng = jax.random.split(rng)
    _, f_init_params = f_init_fun(rng_f, input_shape = (BATCH_SIZE, X_SIZE))
    
    q_opt_state = q_init(q_init_params)
    f_opt_state = f_init(f_init_params)

    #@jax.jit
    def q_opt_update(cnt, q_opt_state, f_opt_state, xx_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        loss_val, grad_val = jax.value_and_grad(q_loss, argnums = 0)(q_params, f_params, x_batch, rng)
        return loss_val, q_update(cnt, grad_val, q_opt_state)
    #@jax.jit
    def f_opt_update(cnt, q_opt_state, f_opt_state, x_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        loss_val, grad_val = jax.value_and_grad(f_loss, argnums = 1)(q_params, f_params, x_batch, rng)
        return loss_val, f_update(cnt, grad_val, f_opt_state)
    def LSDE(   q_params,
                f_params,
                x,  # NO batch
                rng):
        assert(x.ndim == 1)
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_f_fun = jax.jacfwd(f_apply_fun, argnums = 1)

        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.shape == (x.size,))
        f_val = f_apply_fun(f_params, x)
        assert(f_val.shape == (x.size,))
        term1 = jnp.dot(grad_x_log_q_val, f_val)    # shape (n,) and shape (n, )
        assert(term1.size == 1) # scalar

        epsilon = jax.random.normal(rng, (x.size, 1))
        grad_x_f_val = grad_x_f_fun(f_params, x)
        assert(grad_x_f_val.shape == (x.size, x.size))
        dot12 = jnp.dot(epsilon.T, grad_x_f_val)
        assert(dot12.shape == (1, x.size))
        dot123 = jnp.dot(dot12, epsilon)
        assert(dot123.size == 1)    # scalar
        lsde = term1 + dot123
        assert(lsde.size == 1)    # scalar
        lsde = lsde.sum()
        return lsde, f_val
    def f_loss(q_params, f_params, x_batch, rng):
        target = 0.0
        for x in x_batch:
            lsde, f_val = LSDE(q_params, f_params, x, rng)
            assert(f_val.shape == (x.size,))
            target += lsde - LAMBDA * (f_val ** 2).sum()
        target /= x_batch.shape[0]
        return -target # flip sign to maximize
    def q_loss(q_params, f_params, x_batch, rng):
        target = 0.0
        for x in x_batch:
            lsde, _ = LSDE(q_params, f_params, jnp.ones((2,)), rng)
            target += lsde
        target /= x_batch.shape[0]
        return target
    
    T = 1000
    C = 5
    sampler = Sampler(BATCH_SIZE)
    c_cnt = 0
    t_cnt = 0

    t0 = time.time()
    q_loss_val, f_loss_val = 0.0, 0.0
    for t in range(T):
        for c in range(C):
            x_batch = sampler.sample()   # batch=1
            assert(x_batch.shape == (BATCH_SIZE, X_SIZE))
            rng1, rng = jax.random.split(rng)
            f_loss_val, f_opt_state = f_opt_update(c_cnt, q_opt_state, f_opt_state, x_batch, rng1)
            c_cnt += 1
            t1 = time.time()
            print(t, "{:.1f}sec".format((t1 - t0) * 1000), q_loss_val, f_loss_val)
            t0 = t1
        x_batch = sampler.sample()
        rng1, rng = jax.random.split(rng)
        q_loss_val, q_opt_state = q_opt_update(t_cnt, q_opt_state, f_opt_state, x_batch, rng1)
        t_cnt += 1

def trial():
    def f(p, x):
        return jax.numpy.asarray(
        [   p * 1 * x,
            p * 2 * x,
            p * 3 * x,
            p * 4 * x]).T
    x = np.array([[1., 2., 3.],[1., 2., 3.]])
    jf1 = jax.jacfwd(f, argnums = 1)
    jf2 = jax.jacrev(f, argnums = 1)
    ans1 = jf1(1, x)
    ans2 = jf2(1, x)
    print(x.shape)
    print(f(1, x).shape)
    print(ans1.shape)
    print(ans2.shape)
    print((ans1 == ans2).all())

if __name__ == "__main__":
    main()
    print("Done.")
