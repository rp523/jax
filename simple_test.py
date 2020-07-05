#coding: utf-8
import os, time, pickle, argparse
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.experimental.stax import serial, parallel, Dense, Sigmoid, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental import optimizers

from model.maker.model_maker import net_maker
from ebm.sampler import Sampler

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

def Q_Net(scale):
    unit_num = 1000
    net = net_maker()
    net.add_layer(serial(Dense(unit_num), Swish()))
    for _ in range(10):
        net.add_layer(serial(SkipDense(unit_num), Swish()))
    net.add_layer(serial(Dense(1)), name = "raw")
    net.add_layer(ProdGaussian(scale), name = "out", input_name = ("raw", None))
    return net.get_jax_model()

def F_Net(scale):
    unit_num = 1000
    net = net_maker()
    net.add_layer(serial(Dense(unit_num), Swish()))
    for _ in range(10):
        net.add_layer(serial(SkipDense(unit_num), Swish()))
    net.add_layer(serial(Dense(2)), name = "out")
    return net.get_jax_model()

def tgt_fun(x):
    return Sampler.prob(x) * 1E-2

def main(is_training):
    LR = 1E-6
    LAMBDA = 0.5
    BATCH_SIZE = 4
    X_DIM = 2
    SAVE_PATH = "simple.bin"
    half = 15
    band = half * 2

    q_init_fun, q_apply_fun_raw = Q_Net(5)
    f_init_fun, f_apply_fun_raw = F_Net(5)
    def q_apply_fun(q_params, x):
        ret = q_apply_fun_raw(q_params, x)["out"]
        if (ret.size == 1):
            ret = ret.sum()
        return ret
    def f_apply_fun(f_params, x, q_params):
        return exact_critic(q_params, f_params, x)
    q_opt_init, q_opt_update, q_get_params = optimizers.adam(LR, b1=0.5, b2=0.9)
    f_opt_init, f_opt_update, f_get_params = optimizers.adam(LR, b1=0.5, b2=0.9)
    
    rng = jax.random.PRNGKey(0)
    rng_s, rng = jax.random.split(rng)
    sampler = Sampler(rng_s, BATCH_SIZE, half)
    if os.path.exists(SAVE_PATH):
        q_init_params, f_init_params = pickle.load(open(SAVE_PATH, "rb"))
        print("LOADED INIT PARAMS")
    else:
        rng_q, rng_f, rng = jax.random.split(rng, 3)
        _, q_init_params = q_init_fun(rng_q, (BATCH_SIZE, X_DIM))
        _, f_init_params = f_init_fun(rng_f, (BATCH_SIZE, X_DIM))
    q_opt_state = q_opt_init(q_init_params)
    f_opt_state = f_opt_init(f_init_params)

    def q_loss(q_params, f_params):
        if 0:
            loss = 0.0
            x_batch = sampler.sample()
            for x in x_batch:
                lsd, _ = LSD(q_params, f_params, x, rng)
                loss += lsd
            loss /= x_batch.shape[0]
        else:
            x = jax.random.uniform(rng, (BATCH_SIZE, X_DIM)) * band - half
            p = q_apply_fun(q_params, x)
            y = tgt_fun(x).reshape((-1, 1))
            def smooth_l1(x):
                return (0.5 * x ** 2) * (jnp.abs(x) < 1) + (jnp.abs(x) - 0.5) * (jnp.abs(x) >= 1)
            assert(p.shape == y.shape)
            loss = smooth_l1(p - y).sum() / x.shape[0]
        return loss
    def exact_critic(   q_params,
                        f_params,
                        x,  # NO batch
                        ):
        assert(x.ndim == 1)
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.shape == (x.size,))

        def log_p(x):
            assert(x.ndim == 1)
            x = x.reshape((1, x.size))  # reshape of 1-batch
            p = Sampler.prob(x)
            if p.size == 1:
                p = p.sum()
            return jnp.log(p)
        grad_x_log_p_fun = jax.grad(log_p)
        grad_x_log_p_val = grad_x_log_p_fun(x)
        
        return (grad_x_log_q_val - grad_x_log_p_val) / (2 * LAMBDA)
    def LSD(    q_params,
                f_params,
                x,  # NO batch
                rng):
        assert(x.ndim == 1)
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.shape == (x.size,))
        f_val = f_apply_fun(f_params, x, q_params)
        assert(f_val.shape == (x.size,))
        term1 = jnp.dot(grad_x_log_q_val, f_val)    # shape (n,) and shape (n, )
        assert(term1.size == 1) # scalar
        
        def EfficientTrace(rng, x, q_params, f_params):
            grad_x_f_fun = jax.jacfwd(f_apply_fun, argnums = 1)
            epsilon = jax.random.normal(rng, (x.size,))
            grad_x_f_val = grad_x_f_fun(f_params, x)
            assert(grad_x_f_val.shape == (x.size, x.size))
            dot12 = jnp.dot(epsilon, grad_x_f_val)
            assert(dot12.shape == (x.size,))
            dot123 = jnp.dot(dot12, epsilon)
            assert(dot123.size == 1)    # scalar
            return dot123.sum()
        
        def Trace(rng, x, q_params, f_params):
            grad_x_f_fun = jax.jacfwd(f_apply_fun, argnums = 1)
            grad_x_f_val = grad_x_f_fun(f_params, x, q_params)
            assert(grad_x_f_val.shape == (x.size, x.size))
            return (grad_x_f_val * jnp.eye(x.size)).sum()

        trace = Trace(rng, x, q_params, f_params)
        lsd = term1 + trace
        assert(lsd.size == 1)    # scalar
        lsd = lsd.sum()
        return lsd, f_val

    @jax.jit
    def q_update(i, q_opt_state, f_opt_state, rng):
        q_params = q_get_params(q_opt_state)
        f_params = f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(q_loss, argnums = 0)(q_params, f_params)
        return loss_val, q_opt_update(i, grad_val, q_opt_state)
    
    def save_img(q_params):
        bin_num = 100
        plot_band = half * 1
        x = jnp.linspace(-plot_band, plot_band, bin_num)
        x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
        y = jnp.linspace(-plot_band, plot_band, bin_num)
        y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
        data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
        assert(data.shape == (bin_num * bin_num, 2))
        minus_E = q_apply_fun(q_params, data)
        #minus_E = tgt_fun(data)
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
        rng_q, rng_f, rng = jax.random.split(rng, 3)
        loss_val, q_opt_state = q_update(e, q_opt_state, f_opt_state, rng_q)
        t1 = time.time()
        if t1 - t0 > 1:
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))
            save_img(q_get_params(q_opt_state))
            print(e, loss_val, t1 - t0)
            t0 = t1
        e += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")