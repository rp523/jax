import os, time, pickle
from matplotlib import pyplot as plt
import jax
from jax.experimental.stax import serial, Dense, elementwise, FanOut, FanInSum, parallel, Identity, Tanh
import jax.experimental.optimizers as optimizers
import jax.numpy as jnp
from ebm.toy_sampler import Sampler
from ebm.lsd import LSD_Learner
from model.maker.model_maker import net_maker
import hydra

SEED = 0
X_DIM = 2
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

def save_map(   q_apply_fun, q_params,
                f_apply_fun, f_params,
                sampler, half_band,
                save_path):
    x = jnp.linspace(-half_band, half_band, bin_num)
    x = jnp.tile(x.reshape(1, -1), (bin_num, 1))
    y = jnp.linspace(-half_band, half_band, bin_num)
    y = jnp.tile(y.reshape(-1, 1), (1, bin_num))
    data = jnp.append(x.reshape(-1, 1), y.reshape(-1, 1), axis = 1)
    assert(data.shape == (bin_num * bin_num, 2))

    log_q = q_apply_fun(q_params, data)
    q = jnp.exp(log_q).reshape((bin_num, bin_num))
    fx = f_apply_fun(f_params, data).reshape((bin_num, bin_num, 2))

    plt.clf()
    fig = plt.figure(figsize=(12, 10))
    
    X = jnp.linspace(-half_band, half_band, bin_num)
    Y = jnp.linspace(-half_band, half_band, bin_num)
    X, Y = jnp.meshgrid(X, Y)

    ax = fig.add_subplot(221)
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

@hydra.main(config_path="lsd_toy.yaml")
def main(cfg):
    print(cfg.pretty())

    _rng = jax.random.PRNGKey(SEED)
    
    _rng, rng_s, rng_q, rng_f = jax.random.split(_rng, 4)
    sampler = Sampler(rng_s, cfg.optim.batch_size * (cfg.optim.critic_loop + 1), cfg.data.half_band)
    mu, sigma = LSD_Learner.get_scale(sampler, 1000, X_DIM)

    q_init_fun, q_apply_fun = LSD_Learner.gaussian_net(mlp(1), mu, sigma)
    f_init_fun, f_apply_fun_raw = mlp(2)
    def f_apply_fun(f_params, x):
        return f_apply_fun_raw(f_params, x)["out"]
    q_opt_init, q_opt_update, q_get_params = \
        optimizers.adam(cfg.optim.lr)
    f_opt_init, f_opt_update, f_get_params = \
        optimizers.adam(cfg.optim.lr)

    @jax.jit
    def update( t_cnt, c_cnt,
                q_opt_state, f_opt_state, x_batch,
                rng):
        idx = 0

        rngs = jax.random.split(rng, (1 + cfg.optim.critic_loop) + 1)
        q_opt_state, q_loss_val = LSD_Learner.q_update(t_cnt, q_opt_state, f_opt_state, x_batch[idx * cfg.optim.batch_size : (idx+1) * cfg.optim.batch_size],
                                                q_apply_fun, f_apply_fun, q_get_params, f_get_params, q_opt_update, rngs[idx])
        idx += 1
        t_cnt += 1
        for _ in range(cfg.optim.critic_loop):
            c_cnt, f_opt_state, f_loss_val = LSD_Learner.f_update( c_cnt, q_opt_state, f_opt_state, x_batch[idx * cfg.optim.batch_size : (idx+1) * cfg.optim.batch_size], cfg.optim.critic_l2,
                                                q_apply_fun, f_apply_fun, q_get_params, f_get_params, f_opt_update, rngs[idx])
            idx += 1
        return t_cnt, c_cnt, q_opt_state, f_opt_state, q_loss_val, f_loss_val, rngs[idx]

    SAVE_PATH = r"params.bin"
    if not os.path.exists(SAVE_PATH):
        _, q_init_params = q_init_fun(rng_q, (cfg.optim.batch_size, X_DIM))
        _, f_init_params = f_init_fun(rng_f, (cfg.optim.batch_size, X_DIM))
    else:
        (q_init_params, f_init_params) = pickle.load(open(SAVE_PATH, "rb"))
        print("LODADED INIT WEIGHT")
    q_opt_state = q_opt_init(q_init_params)
    f_opt_state = f_opt_init(f_init_params)

    t0 = time.time()
    t = c = 0
    q_loss_val = f_loss_val = 0.0

    while t < cfg.optim.loop_num:
        x_batch = sampler.sample()

        t, c, q_opt_state, f_opt_state, q_loss_val, f_loss_val, _rng \
            = update(t, c, q_opt_state, f_opt_state, x_batch, _rng)

        t1 = time.time()
        if t1 - t0 > 20.0:
            print_txt = ""
            for txt in ["{}".format(t),
                        "{:.2f}".format(t1 - t0),
                        "{:.6f}".format(q_loss_val),
                        "{:.6f}".format(-f_loss_val),
                        ]:
                if print_txt != "":
                    print_txt += ","
                print_txt += txt
            print(print_txt)
            with open("learn_log.txt", "a") as f:
                f.write("{}".format(print_txt))
            t0 = t1
            save_map(   q_apply_fun, q_get_params(q_opt_state),
                        f_apply_fun, f_get_params(f_opt_state),
                        sampler, cfg.data.half_band,
                        "map.png")
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))

if __name__ == "__main__":
    main()
    print("Done.")
