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
    rng = jax.random.PRNGKey(0)
    q_init_fun, q_apply_fun_ = mlp(1)
    f_init_fun, f_apply_fun_ = mlp(2)
    def q_apply_fun(params, x):
        return q_apply_fun_(params, x)["out"][0]    # batch=1
    def f_apply_fun(params, x):
        return f_apply_fun_(params, x)["out"][0]    # batch=1
    LR = 1E-3
    LAMBDA = 0.5
    q_init, q_update, q_get_params = optimizers.adam(LR)
    f_init, f_update, f_get_params = optimizers.adam(LR)

    rng1, rng = jax.random.split(rng)
    _, q_init_params = q_init_fun(rng1, (2, ))
    rng1, rng = jax.random.split(rng)
    _, f_init_params = f_init_fun(rng1, (2, ))
    
    q_opt_state = q_init(q_init_params)
    f_opt_state = f_init(f_init_params)

    @jax.jit
    def q_opt_update(cnt, q_opt_state, f_opt_state, xx_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        grad_val = jax.grad(q_loss, argnums = 0)(q_params, f_params, x_batch, rng)
        return q_update(cnt, grad_val, q_opt_state)
    @jax.jit
    def f_opt_update(cnt, q_opt_state, f_opt_state, x_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        grad_val = jax.grad(f_loss, argnums = 1)(q_params, f_params, x_batch, rng)
        return f_update(cnt, grad_val, f_opt_state)
    def LSDE(q_params, f_params, x, rng):
        grad_x_log_q = jax.grad(jax.grad(q_apply_fun, argnums = 1))
        grad_x_f = jax.jacfwd(f_apply_fun)
        assert(x_batch.ndim == 1)
        x_dim = x_batch.shape[0]

        grad_x_log_q_val = grad_x_log_q(q_params, x)
        assert(grad_x_log_q_val.shape == (batch_size, x_dim))
        f_val = f_apply_fun(f_params, x)["out"]
        assert(f_val.shape == (batch_size, x_dim))
        term1 = jnp.dot(grad_x_log_q_val.T, f_val)
        assert(term1.shape == (batch_size,))
        term1 = term1.mean()

        epsilon = jax.random.normal(rng, (x_dim, batch_size))
        grad_x_f_val = grad_x_f(f_params, x)
        assert(grad_x_f_val.shape == (batch_size, x_dim, x_dim))
        dot12 = jnp.dot(epsilon.T, grad_x_f_val.transpose(1, 2, 3))
        assert(dot12.shape == (batch_size, x_dim))
        dot123 = jnp.dot(dot12, epsilon)
        assert(dot123.shape == (batch_size))
        term2 = dot123.mean()
        return (term1 + term2), fval
    def f_loss(q_params, f_params, x_batch, rng):
        lsde, f_val = LSDE(q_params, f_params, x_batch, rng)
        assert(f_val.shape == (batch_size, x_dim))
        target = lsde - LAMBDA * jnp.dot(f_val, f_val.T).mean()
        return -target # flip sign to maximize
    def q_loss(q_params, f_params, x_batch, rng):
        lsde, _ = LSDE(q_params, f_params, x_batch, rng)
        return target
    
    BATCH_SIZE = 1
    T = 1000
    C = 5
    sampler = Sampler(BATCH_SIZE)
    c_cnt = 0
    t_cnt = 0

    t0 = time.time()
    for t in range(T):
        for c in range(C):
            x_batch = sampler.sample()[0]   # batch=1
            rng1, rng = jax.random.split(rng)
            f_opt_state = f_opt_update(c_cnt, q_opt_state, f_opt_state, x_batch, rng1)
            c_cnt += 1
        x_batch = sampler.sample()
        rng1, rng = jax.random.split(rng)
        q_opt_state = q_opt_update(t_cnt, q_opt_state, f_opt_state, x_batch, rng1)
        t_cnt += 1
        t1 = time.time()
        print(t, "{.02f}sec".format((t1 - t0) / 1000))
        t0 = t1

if __name__ == "__main__":
    main()
    print("Done.")