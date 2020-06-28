#coding: utf-8
import os, time, pickle, argparse
import numpy as np
import jax
from matplotlib import pyplot as plt
from jax import numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import  (serial, parallel, Dense, Tanh, elementwise, BatchNorm, Identity,
                                    FanInSum, FanOut, Sigmoid, Relu)
from ebm.sampler import Sampler
from model.maker.model_maker import net_maker

def swish(x):
    return x / (1.0 + jnp.exp(- x))
Swish = elementwise(swish)

def mlp(out_ch):
    net = net_maker()
    for _ in range(2):
        net.add_layer(serial(Dense(300), Relu))
    net.add_layer(serial(Dense(out_ch)), name = "out")
    return net.get_jax_model()

def main(is_training):
    LR = 1E-3
    LAMBDA = 0.5
    BATCH_SIZE = 8
    X_SIZE = 2
    C = 0

    q_init_fun, q_apply_fun_raw = mlp(1)
    f_init_fun, f_apply_fun_raw = mlp(X_SIZE)
    def q_apply_fun(params, x):
        ret = q_apply_fun_raw(params, x)["out"]
        if ret.size == 1:
            ret = ret.sum()
        return ret
    # NN-approximate critic 
    def f_apply_fun_(f_params, x, q_params):
        return f_apply_fun_raw(f_params, x)["out"]
    # exact critic
    def f_apply_fun(f_params, x, q_params):
        return exact_critic(q_params, f_params, x)

    q_init, q_update, q_get_params = optimizers.adam(LR)
    f_init, f_update, f_get_params = optimizers.adam(LR)

    def exact_critic(   q_params,
                        f_params,
                        x,  # NO batch
                        ):
        assert(x.ndim == 1)
        grad_x_log_q_fun = jax.grad(q_apply_fun, argnums = 1)
        grad_x_log_q_val = grad_x_log_q_fun(q_params, x)
        assert(grad_x_log_q_val.shape == (x.size,))

        def log_p(x):
            p = Sampler.prob(x)
            return jnp.log(p)
        grad_x_log_p_fun = jax.grad(log_p)
        grad_x_log_p_val = grad_x_log_p_fun(x)
        
        return (grad_x_log_q_val - grad_x_log_p_val) / (2 * LAMBDA)

    @jax.jit
    def q_opt_update(cnt, q_opt_state, f_opt_state, x_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        loss_val, grad_val = jax.value_and_grad(q_loss_d, argnums = 0)(q_params, f_params, x_batch, rng)
        return loss_val, q_update(cnt, grad_val, q_opt_state)
    @jax.jit
    def f_opt_update(cnt, q_opt_state, f_opt_state, x_batch, rng):
        f_params = f_get_params(f_opt_state)
        q_params = q_get_params(q_opt_state)
        loss_val, grad_val = jax.value_and_grad(f_loss, argnums = 1)(q_params, f_params, x_batch, rng)
        return loss_val, f_update(cnt, grad_val, f_opt_state)
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
    def f_loss(q_params, f_params, x_batch, rng):
        target = 0.0
        for x in x_batch:
            lsd, f_val = LSD(q_params, f_params, x, rng)
            assert(f_val.shape == (x.size,))
            target += (lsd - LAMBDA * (f_val ** 2).sum())
        target /= x_batch.shape[0]
        return -target # flip sign to maximize
    # generative approach
    def q_loss(q_params, f_params, x_batch, rng):
        target = 0.0
        for x in x_batch:
            lsd, _ = LSD(q_params, f_params, x, rng)
            target += lsd
        target /= x_batch.shape[0]
        target += 1E-4 * net_maker.weight_decay(q_params)
        return target
    
    # discrimitive approach
    def q_loss_d(q_params, f_params, x_batch_biased, rng):
        x_batch = jax.random.uniform(rng, (BATCH_SIZE, X_SIZE)) - 0.5
        
        pred = q_apply_fun(q_params, x_batch)
        #pred = jax.nn.sigmoid(pred) * 1.5295591
        #assert(pred.min() >= 0)
        #assert(pred.max() <= 1.5295591)
        tgt  = Sampler.prob(x_batch)
        #assert(tgt.min() >= 0)
        #assert(tgt.max() <= 1.5295591)
        loss = (pred - tgt) ** 2
        loss = loss.sum() / x_batch.shape[0]

        #loss += 1E-4 * net_maker.weight_decay(q_params)
        return loss
    
    rng = jax.random.PRNGKey(0)
    sampler = Sampler(rng, BATCH_SIZE)
    c_cnt = 0
    t_cnt = 0

    t0 = time.time()
    save_t0 = t0
    q_loss_val, f_loss_val = 0.0, 0.0

    save_dir = "ebm_weight"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    q_param_path = os.path.join(save_dir, "q.pickle")
    f_param_path = os.path.join(save_dir, "f.pickle")
    if os.path.exists(q_param_path) and os.path.exists(f_param_path):
        q_init_params = pickle.load(open(q_param_path, "rb"))
        f_init_params = pickle.load(open(f_param_path, "rb"))
        print("LOADED INIT PARAMS")
    else:
        rng_q, rng_f, rng = jax.random.split(rng, 3)
        _, q_init_params = q_init_fun(rng_q, input_shape = (BATCH_SIZE, X_SIZE))
        rng1, rng = jax.random.split(rng)
        _, f_init_params = f_init_fun(rng_f, input_shape = (BATCH_SIZE, X_SIZE))
    q_opt_state = q_init(q_init_params)
    f_opt_state = f_init(f_init_params)
    
    while is_training:
        for c in range(C):
            x_batch = sampler.sample()   # batch=1
            assert(x_batch.shape == (BATCH_SIZE, X_SIZE))
            rng1, rng = jax.random.split(rng)
            f_loss_val, f_opt_state = f_opt_update(c_cnt, q_opt_state, f_opt_state, x_batch, rng1)
            c_cnt += 1

        x_batch = sampler.sample()
        rng1, rng = jax.random.split(rng)
        q_loss_val, q_opt_state = q_opt_update(t_cnt, q_opt_state, f_opt_state, x_batch, rng1)
        t_cnt += 1

        t1 = time.time()
        if t1 - t0 > 1:
            print(  t_cnt,
                    "{:.1f}ms".format((t1 - t0) * 1000),
                    q_loss_val,
                    - f_loss_val,
                    )
            t0 = t1

        save_t1 = time.time()
        if (save_t1 - save_t0 > 10):
            save_t0 = save_t1
            with open(q_param_path, "wb") as f:
                pickle.dump(q_get_params(q_opt_state), f)
            with open(f_param_path, "wb") as f:
                pickle.dump(f_get_params(f_opt_state), f)
    
    bin_num = 100
    x = jnp.linspace(-0.5, 0.5, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-0.5, 0.5, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))
    minus_E = q_apply_fun(q_init_params, data)
    assert(minus_E.shape == (bin_num * bin_num, 1))
    unnorm_log_q = minus_E
    unnorm_log_q = unnorm_log_q.reshape((bin_num, bin_num))
    X = jnp.linspace(-0.5, 0.5, bin_num)
    Y = jnp.linspace(-0.5, 0.5, bin_num)
    X, Y = jnp.meshgrid(X, Y)
    plt.pcolor(X, Y, unnorm_log_q)
    plt.colorbar()
    plt.show()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type = int)
    args = parser.parse_args()

    is_training = (args.training != 0)
    main(is_training)
    print("Done.")
