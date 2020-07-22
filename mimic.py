#coding: utf-8
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Dense, Conv, Relu, Tanh
import jax.experimental.optimizers as optimizers
import jax.random as jrandom
import time, os, argparse
from model.maker.model_maker import net_maker
from dataset.mnist import Mnist

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

def classifyer(class_num):
    return serial(  serial( Dense(300), Swish(),
                            Dense(300), Swish(),
                            Dense(300), Swish(),
                            Dense(class_num)))
def estimator():
    return serial(  serial( Dense(300), Swish(),
                            Dense(300), Swish(),
                            Dense(300), Swish(),
                            Dense(1)))
def main(is_learn):
    seed = 0
    batch_size = 128
    c_lr = 1E-5
    e_lr = 1E-3
    learn_class = 9
    focal_gamma = 2.0
    remove_class = 9

    rng = jrandom.PRNGKey(seed)

    c_init_fun, c_apply_fun = classifyer(learn_class)
    e_init_fun, e_apply_fun = estimator()
    c_opt_init, c_opt_update, c_get_params = optimizers.adam(c_lr)
    e_opt_init, e_opt_update, e_get_params = optimizers.adam(e_lr)

    rng, rng_c, rng_e = jrandom.split(rng, 3)
    _, c_init_params = c_init_fun(rng_c, (batch_size, 28 * 28))
    _, e_init_params = e_init_fun(rng_e, (batch_size, 28 * 28))
    rng, rng_l, rng_l, rng_ep, rng_en = jrandom.split(rng, 5)
    learn_sampler    = Mnist(rng_l,  batch_size, "train", one_hot = True , dequantize = True, flatten = True, class_num = learn_class, remove_classes = [remove_class])
    #learn_sampler2   = Mnist(rng_l,  batch_size, "train", one_hot = False, dequantize = True, flatten = True, class_num = learn_class, remove_classes = [remove_class])
    eval_pos_sampler = Mnist(rng_ep, 1000, "test" , one_hot = False, dequantize = True, flatten = True, class_num = learn_class, remove_classes = [remove_class])
    eval_neg_sampler = Mnist(rng_en, batch_size, "test" , one_hot = False, dequantize = True, flatten = True, class_num = 1,           remove_classes = jnp.arange(learn_class))
    
    c_opt_state = c_opt_init(c_init_params)
    e_opt_state = e_opt_init(e_init_params)

    def c_loss(c_params, x, y, gamma):
        logit = c_apply_fun(c_params, x)
        prob = jax.nn.softmax(logit)
        cross_entropy = (- y * ((1.0 - prob) ** gamma) * jnp.log(prob + 1E-10)).sum(axis = -1).mean()
        wd = 1E-5 * net_maker.weight_decay(c_params)
        return cross_entropy + wd
    def e_loss(e_params, c_params, x, y):
        logit = c_apply_fun(c_params, x)
        assert(logit.shape == y.shape)
        label_logit = (logit * y).sum(axis = -1)
        mimic_logit = e_apply_fun(e_params, x).flatten()
        assert(mimic_logit.shape == (batch_size,))
        assert(label_logit.shape == (batch_size,))
        logit_error = jnp.abs(mimic_logit - label_logit)
        return (- jnp.log(logit_error + 1E-10)).mean()
    @jax.jit
    def c_update(cnt, c_opt_state, x, y, gamma):
        c_params = c_get_params(c_opt_state)
        loss_val, grad_val = jax.value_and_grad(c_loss)(c_params, x, y, gamma)
        c_opt_state = c_opt_update(cnt, grad_val, c_opt_state)
        return cnt + 1, loss_val, c_opt_state
    @jax.jit
    def e_update(cnt, e_opt_state, c_opt_state, x, y):
        e_params = e_get_params(e_opt_state)
        c_params = c_get_params(c_opt_state)
        loss_val, grad_val = jax.value_and_grad(e_loss)(e_params, c_params, x, y)
        e_opt_state = e_opt_update(cnt, grad_val, e_opt_state)
        return cnt + 1, loss_val, e_opt_state
    def learn_accuracy(c_opt_state):
        c_params = c_get_params(c_opt_state)
        x, y = eval_pos_sampler.sample(get_all = True)
        logit = c_apply_fun(c_params, x)
        pred_y = logit.argmax(axis = -1)
        return (y == pred_y).mean()
    def ood_rate(e_opt_state, rng):
        e_params = e_get_params(e_opt_state)
        xp, yp = eval_pos_sampler.sample()
        assert(xp.shape == (1000, 28*28))
        assert((yp != remove_class).all())
        xn, yn = eval_neg_sampler.sample(get_all = True)
        assert((yn == remove_class).all())
        mimic_logit_p = e_apply_fun(e_params, xp)
        mimic_logit_n = e_apply_fun(e_params, xn)
        assert(mimic_logit_p.shape[1] == 1)
        assert(mimic_logit_n.shape[1] == 1)
        mimic_logit_p = mimic_logit_p.flatten()
        mimic_logit_n = mimic_logit_n.flatten()
        mimic_logit_all = jnp.append(mimic_logit_p, mimic_logit_n)
        ret = []
        ranres = jrandom.uniform(rng, mimic_logit_all.shape) * (mimic_logit_all.max() - mimic_logit_all.min()) + mimic_logit_all.min()
        assert(ranres.shape == mimic_logit_all.shape)
        assert(ranres.min() >= mimic_logit_all.min())
        assert(ranres.max() <= mimic_logit_all.max())
        for mimip, mimin in [   (mimic_logit_p, mimic_logit_n),
                                (ranres[:mimic_logit_p.size], ranres[mimic_logit_p.size:])]:
            max_rate = 0.0
            for th in ranres:
                p_correct_num = (mimip >= th).sum()
                n_correct_num = (mimin <  th).sum()
                rate = (p_correct_num + n_correct_num) / (mimip.size + mimin.size)
                max_rate = max(rate, max_rate)
            ret.append(float(max_rate))
        return ret[0] - ret[1]
    e = c = 0
    t0 = time.time()
    while is_learn:
        #_x, _y = learn_sampler2.sample()
        #assert((_y != remove_class).all())
        x, y = learn_sampler.sample()
        assert(y.shape == (batch_size, learn_class))
        c, c_loss_val, c_opt_state = c_update(c, c_opt_state, x, y, focal_gamma)
        e, e_loss_val, e_opt_state = e_update(e, e_opt_state, c_opt_state, x, y)

        t1 = time.time()
        if t1 - t0 > 20:
            rng, rng1 = jrandom.split(rng)
            print(  e, 
                    "{:.2f}%".format(100 * learn_accuracy(c_opt_state)),
                    "{:.2f}%".format(100 * ood_rate(e_opt_state, rng1)),
                    "{:.3f}".format(c_loss_val),
                    "{:.3f}".format(e_loss_val),
            )
            t0 = t1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action = "store_true")
    arg = parser.parse_args()
    main(not arg.eval)
    print("Done.")