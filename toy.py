import os, time, pickle
from matplotlib import pyplot as plt
import jax
from jax.experimental.stax import serial, Dense, elementwise, FanOut, FanInSum, parallel, Identity
import jax.experimental.optimizers as optimizers
import jax.numpy as jnp
from ebm.sampler import Sampler
from model.maker.model_maker import net_maker

SEED = 0
BATCH_SIZE = 100
X_DIM = 2
HALF_BAND = 5.0
LR = 1E-4
B1 = 0.0
B2 = 0.999
LAMBDA = 10
C = 5
bin_num = 100

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

def mlp(out_ch):
    net = net_maker()
    net.add_layer(serial(Dense(300), Swish()))
    for _ in range(3):
        net.add_layer(serial(SkipDense(300), Swish()))
    net.add_layer(Dense(out_ch), name = "out")
    return net.get_jax_model()

def gaussian_net(base_net, init_mu, init_sigma):
    base_init_fun, base_apply_fun = base_net
    def init_fun(rng, input_shape):
        rng_base, rng_mu, rng_sigma = jax.random.split(rng, 3)
        _, base_params = base_init_fun(rng_base, input_shape)
        mu_shape = tuple(input_shape[1:])
        mu = jax.nn.initializers.ones(rng_mu, mu_shape) * init_mu
        log_sigma = jnp.log(jax.nn.initializers.ones(rng_sigma, (1,)) * init_sigma)
        params = list(base_params) + [(mu, log_sigma)]
        output_shape = input_shape
        return output_shape, params
    def apply_fun(params, inputs, **kwargs):
        base_params = params[:-1]
        mu, log_sigma = params[-1]
        mu = mu.reshape(tuple([1] + list(mu.shape)))
        sigma = jnp.exp(log_sigma)
        base_output = base_apply_fun(base_params, inputs)["out"]
        log_gauss = - ((inputs - mu) ** 2).sum(axis = -1) / (2 * sigma ** 2)
        log_gauss = log_gauss.reshape((base_output.shape[0], -1))
        ret = base_output + log_gauss
        return ret
    return init_fun, apply_fun

def save_map(   q_apply_fun, q_params,
                f_apply_fun, f_params,
                x_record, sampler,
                save_path):
    x = jnp.linspace(-HALF_BAND, HALF_BAND, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-HALF_BAND, HALF_BAND, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))

    log_q = q_apply_fun(q_params, data)
    q = jnp.exp(log_q).reshape((bin_num, bin_num))
    fx = f_apply_fun(f_params, data).reshape((bin_num, bin_num, 2))

    plt.clf()
    fig = plt.figure(figsize=(12, 10))
    
    X = jnp.linspace(-HALF_BAND, HALF_BAND, bin_num)
    Y = jnp.linspace(-HALF_BAND, HALF_BAND, bin_num)
    X, Y = jnp.meshgrid(X, Y)

    ax = fig.add_subplot(221)
    if x_record is not None:
        plt.pcolor(X, Y, x_record.reshape(bin_num, bin_num))
    else:
        plt.pcolor(X, Y, sampler.prob(data).reshape(bin_num, bin_num))
    plt.colorbar()

    ax = fig.add_subplot(222)
    plt.pcolor(X, Y, q)
    plt.colorbar()

    for d in range(X_DIM):
        ax = fig.add_subplot(223 + d)
        plt.pcolor(X, Y, fx[:,:,d])
        plt.colorbar()

    plt.savefig(save_path)

def get_scale(sampler):
    x = jnp.empty((0, X_DIM))
    while x.shape[0] <= 1000:
        x = jnp.append(x, sampler.sample(), axis = 0)
    return x.mean(), x.std()

class LSD_Learner:
    @staticmethod
    def calc_sq_batch(q_params, x_batch, arg_q_apply_fun):
        def logq_sum(q_params, x_batch):
            logq_batch = arg_q_apply_fun(q_params, x_batch)
            return logq_batch.sum()
        sq_batch = jax.grad(logq_sum, argnums = 1)(q_params, x_batch) # ▽x(Log(q))
        return sq_batch
    @staticmethod
    def calc_efficient_trace(f_params, x_batch, arg_f_apply_fun, rng):
        eps = jax.random.normal(rng, x_batch.shape)
        fx, vjp_fun = jax.vjp(arg_f_apply_fun, f_params, x_batch)
        dp, dx = vjp_fun(eps)
        return (dx * eps).sum(axis = -1), fx
    @staticmethod
    def calc_exact_trace(f_params, x_batch, arg_f_apply_fun):
        trace_batch = jnp.zeros((BATCH_SIZE,))
        def f_apply_fun_dim(f_params, x_batch, idx):
            return arg_f_apply_fun(f_params, x_batch)[:, idx].sum()
        for d in range(X_DIM):
            trace_comp = jax.grad(f_apply_fun_dim, argnums = 1)(f_params, x_batch, d)[:, d]
            trace_batch += trace_comp
        return trace_batch
    @staticmethod
    def calc_loss_metrics(  q_params, f_params, x_batch,
                            arg_q_apply_fun, arg_f_apply_fun, rng):
        tr_dfdx_batch, fx_batch = LSD_Learner.calc_efficient_trace(f_params, x_batch, arg_f_apply_fun, rng)
        sq_batch = LSD_Learner.calc_sq_batch(q_params, x_batch, arg_q_apply_fun)
        sq_fx_batch = (sq_batch * fx_batch).sum(axis = -1)
        lsd = (sq_fx_batch + tr_dfdx_batch).mean()
        f_norm = (fx_batch * fx_batch).sum(axis = -1).mean()
        return lsd, f_norm
    @staticmethod
    def q_loss( q_params, f_params, x_batch,
                arg_q_apply_fun, arg_f_apply_fun, rng):
        lsd, _ =  LSD_Learner.calc_loss_metrics(q_params, f_params, x_batch,
                                    arg_q_apply_fun, arg_f_apply_fun, rng)
        loss = lsd + 1E-5 * net_maker.weight_decay(q_params)
        return loss
    @staticmethod
    def f_loss( q_params, f_params, x_batch,
                arg_q_apply_fun, arg_f_apply_fun, rng):
        lsd, f_norm =  LSD_Learner.calc_loss_metrics(   q_params, f_params, x_batch,
                                            arg_q_apply_fun, arg_f_apply_fun, rng)
        return -lsd + LAMBDA * f_norm
    @staticmethod
    def q_update(   t_cnt, q_opt_state, f_opt_state, x_batch,
                    arg_q_apply_fun, arg_f_apply_fun, arg_q_get_params, arg_f_get_params, arg_q_opt_update, rng):
        q_params = arg_q_get_params(q_opt_state)
        f_params = arg_f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(LSD_Learner.q_loss, argnums = 0)(q_params, f_params, x_batch, arg_q_apply_fun, arg_f_apply_fun, rng)
        q_opt_state = arg_q_opt_update(t_cnt, grad_val, q_opt_state)
        return q_opt_state, loss_val
    @staticmethod
    def f_update(c_cnt, q_opt_state, f_opt_state, x_batch,
                    arg_q_apply_fun, arg_f_apply_fun, arg_q_get_params, arg_f_get_params, arg_f_opt_update, rng):
        q_params = arg_q_get_params(q_opt_state)
        f_params = arg_f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(LSD_Learner.f_loss, argnums = 1)(q_params, f_params, x_batch, arg_q_apply_fun, arg_f_apply_fun, rng)
        f_opt_state = arg_f_opt_update(c_cnt, grad_val, f_opt_state)
        return f_opt_state, loss_val

def main():
    _rng = jax.random.PRNGKey(SEED)
    
    _rng, rng_s, rng_q, rng_f = jax.random.split(_rng, 4)
    sampler = Sampler(rng_s, BATCH_SIZE * (C + 1), HALF_BAND)
    mu, sigma = get_scale(sampler)

    q_init_fun, q_apply_fun = gaussian_net(mlp(1), mu, sigma)
    f_init_fun, f_apply_fun_raw = mlp(2)
    def f_apply_fun(f_params, x):
        return f_apply_fun_raw(f_params, x)["out"]
    q_opt_init, q_opt_update, q_get_params = \
        optimizers.adam(LR, b1=B1, b2=B2)
    f_opt_init, f_opt_update, f_get_params = \
        optimizers.adam(LR, b1=B1, b2=B2)

    @jax.jit
    def update( t_cnt, c_cnt,
                q_opt_state, f_opt_state, x_batch,
                rng):
        idx = 0

        rngs = jax.random.split(rng, (1 + C) + 1)
        q_opt_state, q_loss_val = LSD_Learner.q_update(t_cnt, q_opt_state, f_opt_state, x_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE],
                                                q_apply_fun, f_apply_fun, q_get_params, f_get_params, q_opt_update, rngs[idx])
        idx += 1
        t_cnt += 1
        for _ in range(C):
            f_opt_state, f_loss_val = LSD_Learner.f_update( c_cnt, q_opt_state, f_opt_state, x_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE],
                                                q_apply_fun, f_apply_fun, q_get_params, f_get_params, f_opt_update, rngs[idx])
            idx += 1
            c_cnt += 1
        return t_cnt, c_cnt, q_opt_state, f_opt_state, q_loss_val, f_loss_val, rngs[idx]

    SAVE_PATH = r"params.bin"
    if not os.path.exists(SAVE_PATH):
        _, q_init_params = q_init_fun(rng_q, (BATCH_SIZE, X_DIM))
        _, f_init_params = f_init_fun(rng_f, (BATCH_SIZE, X_DIM))
    else:
        (q_init_params, f_init_params) = pickle.load(open(SAVE_PATH, "rb"))
        print("LODADED INIT WEIGHT")
    q_opt_state = q_opt_init(q_init_params)
    f_opt_state = f_opt_init(f_init_params)
    x_record = None#jnp.zeros((bin_num, bin_num), dtype = jnp.uint32)
    def update_x_record(x_record, x_batch):
        if x_record is not None:
            for x in x_batch:
                xi0 = jnp.clip(((x[0] / HALF_BAND + 1.0) / 2.0 * bin_num).astype(jnp.int32), 0, bin_num - 1)
                xi1 = jnp.clip(((x[1] / HALF_BAND + 1.0) / 2.0 * bin_num).astype(jnp.int32), 0, bin_num - 1)
                x_record = jax.ops.index_add(x_record, jax.ops.index[xi0, xi1], 1)
        return x_record

    t0 = time.time()
    t = c = 0
    q_loss_val = f_loss_val = 0.0
    olds = f_get_params(f_opt_state)
    while True:
        x_batch = sampler.sample()
        x_record = update_x_record(x_record, x_batch)

        t, c, q_opt_state, f_opt_state, q_loss_val, f_loss_val, _rng \
            = update(t, c, q_opt_state, f_opt_state, x_batch, _rng)

        t1 = time.time()
        if t1 - t0 > 10.0:
            news = f_get_params(f_opt_state)
            print(t, "{:.2f}".format(t1 - t0), q_loss_val, -1 * f_loss_val,
                    net_maker.param_l2_norm(olds, news))
            olds = news
            t0 = t1
            save_map(   q_apply_fun, q_get_params(q_opt_state),
                        f_apply_fun, f_get_params(f_opt_state),
                        x_record, sampler,
                        "map.png")
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))

if __name__ == "__main__":
    main()
    print("Done.")
