#coding: utf-8
import os, time, pickle, argparse
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.experimental.stax import serial, parallel, Dense, Sigmoid, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental import optimizers
from model.maker.model_maker import net_maker
from ebm.sampler import Sampler
MODE = "discriminative"
MODE = "generative"
TRAIN_CRITIC = True

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
        if x.ndim > 1:  # batch
            axis = tuple(jnp.arange(1, x.ndim))
        else:
            axis = None
        negative_energy = -((x - mu) ** 2).sum(axis = axis).reshape(-1, 1) / (2.0 * (sigma ** 2))
        if MODE == "generative":
            output_val = base_val + negative_energy
            output_val = output_val.reshape(base_val.shape)
        elif MODE == "discriminative":
            output_val = base_val * jnp.exp(negative_energy)
        return output_val
    return init_fun, apply_fun

def Q_Net(scale):
    unit_num = 300
    net = net_maker()
    for _ in range(2):
        net.add_layer(Dense(unit_num))
        net.add_layer(Swish())
    net.add_layer(Dense(1), name = "raw")
    net.add_layer(ProdGaussian(scale), name = "out", input_name = ("raw", None))
    return net.get_jax_model()

def F_Net(scale):
    unit_num = 300
    net = net_maker()
    for _ in range(2):
        net.add_layer(Dense(unit_num))
        net.add_layer(Swish())
    net.add_layer(Dense(2), name = "raw")
    net.add_layer(ProdGaussian(scale), name = "out", input_name = ("raw", None))
    return net.get_jax_model()

def tgt_fun(sampler, x ):
    return sampler.prob(x)

def main(is_training):
    RANDOM_SEED = 1
    LR = 1E-3
    LAMBDA = 10
    BATCH_SIZE = 8
    X_DIM = 2
    C = 5
    SAVE_PATH = "simple.bin"
    SAVE_X_DIST = False
    half = 1.0
    band = half * 2
    x_record_bin = 100

    q_init_fun, q_apply_fun_raw = Q_Net(half)
    f_init_fun, f_apply_fun_raw = F_Net(half)
    def q_apply_fun(q_params, x):
        ret = q_apply_fun_raw(q_params, x)["out"]
        if (ret.size == 1):
            ret = ret.sum()
        return ret
    def f_apply_fun(f_params, x, q_params):
        if TRAIN_CRITIC:
            ret = f_apply_fun_raw(f_params, x)["out"]
        else:
            ret = exact_critic(q_params, f_params, x)
        return ret
    q_opt_init, q_opt_update, q_get_params = optimizers.adam(LR, b1=0.5, b2=0.9)
    f_opt_init, f_opt_update, f_get_params = optimizers.adam(LR, b1=0.5, b2=0.9)
    
    rng = jax.random.PRNGKey(RANDOM_SEED)
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

    def q_loss(q_params, f_params, x_batch, rng):
        if MODE == "generative":
            sum_lsd = 0.0
            for x in x_batch:
                x = x.reshape(tuple([1] + list(x.shape))) # make 1-batch shape
                lsd, f_val = LSD(q_params, f_params, x, rng)
                sum_lsd += lsd
            ave_lsd = sum_lsd / BATCH_SIZE
            loss = ave_lsd
            loss += 1E-5 * net_maker.weight_decay(q_params)
            return loss
        elif MODE == "discriminative":
            p = q_apply_fun(q_params, x_batch)
            y = tgt_fun(sampler, x_batch).reshape((-1, 1))
            def smooth_l1(x):
                return (0.5 * x ** 2) * (jnp.abs(x) < 1) + (jnp.abs(x) - 0.5) * (jnp.abs(x) >= 1)
            assert(p.shape == y.shape)
            loss = smooth_l1(p - y).sum() / BATCH_SIZE
        return loss
    
    def f_loss(q_params, f_params, x_batch, rng):
        sum_lsd = 0.0
        sum_f_norm = 0.0
        for x in x_batch:
            x = x.reshape(tuple([1] + list(x.shape))) # make 1-batch shape
            lsd, f_val = LSD(q_params, f_params, x, rng)
            assert(f_val.size == X_DIM)
            sum_lsd += lsd
            sum_f_norm += (f_val ** 2).sum()
            #f_val_ideal = exact_critic(q_params, f_params, x)

        ave_lsd = sum_lsd / BATCH_SIZE
        loss = ave_lsd
        ave_f_norm = sum_f_norm / BATCH_SIZE
        loss -= LAMBDA * ave_f_norm
        #loss -= 1E-5 * net_maker.weight_decay(f_params)
        
        return -loss    # flip sign to maximize

    def exact_critic(   q_params,
                        f_params,
                        x,  # NO batch
                        ):
        assert(x.shape == (1, X_DIM))
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.size == X_DIM)
        grad_x_log_q_val = grad_x_log_q_val.reshape((X_DIM,))

        def log_p(x):
            assert(x.shape == (1, X_DIM))
            p = sampler.prob(x)
            if p.size == 1:
                p = p.sum()
            return jnp.log(p)
        grad_x_log_p_fun = jax.grad(log_p)
        grad_x_log_p_val = grad_x_log_p_fun(x)
        assert(grad_x_log_p_val.size == X_DIM)
        grad_x_log_p_val = grad_x_log_p_val.reshape((X_DIM,))
        
        return (grad_x_log_q_val - grad_x_log_p_val) / (2 * LAMBDA)
    def LSD(    q_params,
                f_params,
                x,  # NO batch
                rng):
        assert(x.shape == (1, X_DIM))
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.shape == (1, X_DIM))
        f_val = f_apply_fun(f_params, x, q_params)
        assert(f_val.size == X_DIM)
        f_val = f_val.reshape((1, X_DIM))
        term1 = (grad_x_log_q_val * f_val).sum()
        
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
            assert(grad_x_f_val.size == (X_DIM * X_DIM))
            return (grad_x_f_val.reshape((X_DIM, X_DIM)) * jnp.eye(x.size)).sum()

        trace = Trace(rng, x, q_params, f_params)
        lsd = term1 + trace
        assert(lsd.size == 1)    # scalar
        lsd = (lsd.sum())
        if not TRAIN_CRITIC:
            lsd = jnp.abs(lsd)
        return lsd, f_val

    @jax.jit
    def q_update(i, q_opt_state, f_opt_state, x_batch, rng):
        q_params = q_get_params(q_opt_state)
        f_params = f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(q_loss, argnums = 0)(q_params, f_params, x_batch, rng)
        return loss_val, q_opt_update(i, grad_val, q_opt_state)
    @jax.jit
    def f_update(i, q_opt_state, f_opt_state, x_batch, rng):
        q_params = q_get_params(q_opt_state)
        f_params = f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(f_loss, argnums = 1)(q_params, f_params, x_batch, rng)
        return loss_val, f_opt_update(i, grad_val, f_opt_state)
    
    def save_img(q_params, x_record):
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
        plt.pcolor(X, Y, jnp.exp(unnorm_log_q))
        plt.colorbar()
        plt.savefig("simple.png")
        
        if SAVE_X_DIST:
            X = jnp.arange(x_record_bin)
            Y = jnp.arange(x_record_bin)
            X, Y = jnp.meshgrid(X, Y)
            plt.clf()
            plt.pcolor(X, Y, x_record)
            plt.colorbar()
            plt.savefig("x_record.png")

    def update_x_record(x_record, x_batch):
        incr_idx = jnp.clip((x_batch / band + 0.5) * x_record_bin, 0, x_record_bin).astype(jnp.int32)
        for b in range(BATCH_SIZE):
            x_record = jax.ops.index_add(x_record, jax.ops.index[incr_idx[b, 0], incr_idx[b, 1]], 1)
        return x_record
    
    x_record = jnp.zeros((x_record_bin, x_record_bin), dtype = jnp.uint32)
    t0 = time.time()
    q_loss_val = f_loss_val = 0.0
    q_cnt = f_cnt = 0
    while is_training:
        rng_q, rng = jax.random.split(rng)
        if (MODE == "generative"):
            x_batch = sampler.sample()
        elif (MODE == "discriminative"):
            x_batch = jax.random.uniform(rng, (BATCH_SIZE, X_DIM)) * band - half
        if SAVE_X_DIST:
            x_record = update_x_record(x_record, x_batch)
                
        q_loss_val, q_opt_state = q_update(q_cnt, q_opt_state, f_opt_state, x_batch, rng_q)
        q_cnt += 1
        if (MODE == "generative") and (TRAIN_CRITIC == True):
            for _ in range(C):
                rng_f, rng = jax.random.split(rng)
                x_batch = sampler.sample()
                if SAVE_X_DIST:
                    x_record = update_x_record(x_record, x_batch)
                f_loss_val, f_opt_state = f_update(f_cnt, q_opt_state, f_opt_state, x_batch, rng_f)
                f_cnt += 1
        t1 = time.time()
        if t1 - t0 > 2.0:
            q_params = q_get_params(q_opt_state)
            f_params = f_get_params(q_opt_state)
            pickle.dump((q_params, f_params), open(SAVE_PATH, "wb"))
            save_img(q_params, x_record)
            print(q_cnt, "{:.2f}".format(t1 - t0),
                    q_loss_val, -1.0 * f_loss_val)
            t0 = t1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()
    is_training = (args.training != 0)
    main(is_training)
    print("Done.")
