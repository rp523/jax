import os, time, pickle
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, Dense, elementwise, FanOut, FanInSum, parallel, Identity
import jax.experimental.optimizers as optimizers
from dataset.mnist import Mnist
from ebm.lsd import LSD_Learner
from model.maker.model_maker import net_maker

SEED = 0
BATCH_SIZE = 100
X_DIM = 28 * 28
CLASS_NUM = 10
Q_LR = 1E-4
F_LR = 1E-4
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
    return serial(  serial(Dense(300), Swish()),
                    serial(SkipDense(300), Swish()),
                    serial(SkipDense(300), Swish()),
                    serial(SkipDense(300), Swish()),
                    serial(Dense(out_ch)),
                    )

def jem(base_net, init_mu, init_sigma):
    base_init_fun, base_apply_fun = base_net
    def init_fun(rng, input_shape):
        rng_base, rng_mu, rng_sigma = jax.random.split(rng, 3)
        _, base_params = base_init_fun(rng_base, input_shape)
        mu_shape = tuple(input_shape[1:])
        mu = jax.nn.initializers.ones(rng_mu, mu_shape) * init_mu
        log_sigma = jnp.log(jax.nn.initializers.ones(rng_sigma, (1,)) * init_sigma)
        params = list(base_params) + [(mu, log_sigma)]
        return None, params
    def apply_fun(params, inputs, **kwargs):
        base_params = params[:-1]
        log_q_xy = base_apply_fun(base_params, inputs)
        log_q_x_base = jax.scipy.special.logsumexp(log_q_xy, axis = 1)

        mu, log_sigma = params[-1]
        mu = mu.reshape(tuple([1] + list(mu.shape)))
        sigma = jnp.exp(log_sigma)
        log_gauss = - ((inputs - mu) ** 2).sum(axis = -1) / (2 * sigma ** 2)
        log_gauss = log_gauss.reshape((log_q_x_base.shape[0], -1))
        log_q_x = log_q_x_base + log_gauss
        return {"class_prob" : jax.nn.softmax(log_q_xy),
                "log_density" : log_q_x}
    return init_fun, apply_fun

def get_scale(sampler, sample_num, x_dim):
    x = jnp.empty((0, x_dim))
    while x.shape[0] <= sample_num:
        x_new, _ = sampler.sample()
        x = jnp.append(x, x_new.reshape((x_new.shape[0], -1)), axis = 0)
    return x.mean(), x.std()

def main():
    _rng = jax.random.PRNGKey(SEED)
    
    _rng, rng_d, rng_q, rng_f = jax.random.split(_rng, 4)
    train_sampler = Mnist(rng_d, BATCH_SIZE * (1 + 1 + C), "train", one_hot = True, dequantize = True)
    test_sampler = Mnist(rng_d, 10000, "test", one_hot = False, dequantize = True)
    mu, sigma = get_scale(train_sampler, 1000, X_DIM)
    print("mu={}, sigma={}".format(mu, sigma))

    q_init_fun, q_apply_fun_raw = jem(mlp(CLASS_NUM), mu, sigma)
    f_init_fun, f_apply_fun = mlp(X_DIM)
    q_opt_init, q_opt_update, q_get_params = optimizers.adam(Q_LR)
    f_opt_init, f_opt_update, f_get_params = optimizers.adam(F_LR)

    def q_apply_fun_classify(q_params, x_batch):
        return q_apply_fun_raw(q_params, x_batch)["class_prob"]
    def q_apply_fun_density(q_params, x_batch):
        a = q_apply_fun_raw(q_params, x_batch)
        assert("log_density" in a.keys())
        return a["log_density"]
    def classify_loss(q_params, x_batch, y_batch):
        y_pred = q_apply_fun_classify(q_params, x_batch)
        cross_entropy = (- y_batch * jnp.log(y_pred + 1E-10)).mean()
        weight_loss = net_maker.weight_decay(q_params)
        return (cross_entropy + 1E-5 * weight_loss)
    def classify_update(l_cnt, q_opt_state, x_batch, y_batch):
        q_params = q_get_params(q_opt_state)
        loss_val, grad_val = jax.value_and_grad(classify_loss)(q_params, x_batch, y_batch)
        q_opt_state = q_opt_update(l_cnt, grad_val, q_opt_state)
        return q_opt_state, loss_val
    def accuracy(q_params, arg_test_sampler):
        x_batch, y_batch = arg_test_sampler.sample()
        x_batch = x_batch.reshape((x_batch.shape[0], -1))
        pred_prob = q_apply_fun_classify(q_params, x_batch)
        pred_idx = jnp.argmax(pred_prob, axis = -1)
        return (y_batch == pred_idx).mean()

    @jax.jit
    def update( l_cnt, t_cnt, c_cnt,
                q_opt_state, f_opt_state, x_batch, y_batch,
                rng):
        q_loss_val, f_loss_val, class_loss_val = 0.0, 0.0, 0.0
        idx = 0
        rngs = jax.random.split(rng, (1 + 1 + C))
        
        # learn classifier
        q_opt_state, class_loss_val = classify_update(  l_cnt, q_opt_state,
                                                        x_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE],
                                                        y_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE])
        idx, l_cnt = idx + 1, l_cnt + 1

        if True:
            # learn density
            q_opt_state, q_loss_val = LSD_Learner.q_update(t_cnt, q_opt_state, f_opt_state, x_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE],
                                                    q_apply_fun_density, f_apply_fun, q_get_params, f_get_params, q_opt_update, rngs[idx])
            idx, t_cnt = idx + 1, t_cnt + 1

            # learn cirtic
            for _ in range(C):
                f_opt_state, f_loss_val = LSD_Learner.f_update( c_cnt, q_opt_state, f_opt_state, x_batch[idx * BATCH_SIZE : (idx+1) * BATCH_SIZE], LAMBDA,
                                                    q_apply_fun_density, f_apply_fun, q_get_params, f_get_params, f_opt_update, rngs[idx])
                idx, c_cnt = idx + 1, c_cnt + 1
        
        return l_cnt, t_cnt, c_cnt, q_opt_state, f_opt_state, q_loss_val, f_loss_val, class_loss_val

    SAVE_PATH = r"params.bin"
    if not os.path.exists(SAVE_PATH):
        _, q_init_params = q_init_fun(rng_q, (BATCH_SIZE, X_DIM))
        _, f_init_params = f_init_fun(rng_f, (BATCH_SIZE, X_DIM))
    else:
        (q_init_params, f_init_params) = pickle.load(open(SAVE_PATH, "rb"))
        print("LODADED INIT WEIGHT")
    q_opt_state = q_opt_init(q_init_params)
    f_opt_state = f_opt_init(f_init_params)

    t0 = time.time()
    l = t = c = 0
    q_loss_val = f_loss_val = 0.0
    olds = f_get_params(f_opt_state)
    while True:
        x_batch, y_batch = train_sampler.sample()
        x_batch = x_batch.reshape((x_batch.shape[0], -1))

        rng1, _rng = jax.random.split(_rng)
        l, t, c, q_opt_state, f_opt_state, q_loss_val, f_loss_val, class_loss_val \
            = update(l, t, c, q_opt_state, f_opt_state, x_batch, y_batch, rng1)

        t1 = time.time()
        if t1 - t0 > 10.0:
            news = f_get_params(f_opt_state)
            print(  l,
                    "{:.2f}sec".format(t1 - t0),
                    "{:.2f}%".format(accuracy(q_get_params(q_opt_state), test_sampler) * 100),
                    "qx_loss={:.2f}".format(q_loss_val),
                    "fx_loss={:.2f}".format(-1 * f_loss_val),
                    "qyx_loss={:.2f}".format(class_loss_val),
                    #net_maker.param_l2_norm(olds, news),
                    ) 
            olds = news
            t0 = t1
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))

if __name__ == "__main__":
    main()
    print("Done.")