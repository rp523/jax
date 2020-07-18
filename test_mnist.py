#coding: utf-8
import time
import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, serial, parallel, Relu, Softmax, Tanh, Identity, FanInSum, FanOut, Sigmoid, elementwise
from jax.experimental.optimizers import adam
from model.maker.model_maker import net_maker
from dataset.mnist import Mnist

BATCH_SIZE = 128
LR = 1E-4
DATA_DIM = 28 * 28
CLASS_NUM = 10
INPUT_SHAPE = (BATCH_SIZE, DATA_DIM )
FOCAL_GAMMA = 2.0

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
def MyNet():
    return serial(  Dense(300), Swish(),
                    Dense(300), Swish(),
                    Dense(300), Swish(),
                    Dense(CLASS_NUM), Softmax,
            )

def main():
    rng = jax.random.PRNGKey(0)
    q_init_fun, q_apply_fun = MyNet()

    rng1, rng = jax.random.split(rng)
    _, q_init_params = q_init_fun(rng1, INPUT_SHAPE)

    rng1, rng = jax.random.split(rng)
    sampler = Mnist(rng1, BATCH_SIZE, "train", one_hot = True, dequantize = True, flatten = True)
    test_sampler = Mnist(rng1, BATCH_SIZE, "test", one_hot = False, dequantize = True, flatten = True)

    q_opt_init, q_opt_update, q_get_params = adam(LR)
    q_opt_state = q_opt_init(q_init_params)

    def loss(q_params, arg_x, arg_y):
        assert(arg_x.shape == (BATCH_SIZE, DATA_DIM))
        assert(arg_y.shape == (BATCH_SIZE, CLASS_NUM))
        y_pred = q_apply_fun(q_params, arg_x)
        assert(y_pred.shape == (BATCH_SIZE, CLASS_NUM))
        #assert(y_pred.min() >= 0.0)
        #assert(y_pred.max() <= 1.0)
        cross_entropy = (- arg_y * ((1.0 - y_pred) ** FOCAL_GAMMA) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        #cross_ent = ((y_pred - arg_y) ** 2).sum(axis = -1).mean()
        wd = 1E-1 * net_maker.weight_decay(q_params)
        return (cross_entropy + wd)
    @jax.jit
    def update(arg_t, arg_q_opt_state, arg_x, arg_y):
        q_params = q_get_params(arg_q_opt_state)
        loss_val, grad_val = jax.value_and_grad(loss)(q_params, x, y)
        new_q_opt_state = q_opt_update(arg_t, grad_val, arg_q_opt_state)
        return (arg_t + 1), new_q_opt_state, loss_val
    def accuracy(arg_q_opt_state, arg_test_sampler):
        q_params = q_get_params(arg_q_opt_state)
        x, y = arg_test_sampler.sample(get_all = True)
        _y_pred = q_apply_fun(q_params, x)
        y_pred = _y_pred.argmax(axis = -1)
        assert(y_pred.shape == y.shape)
        return (y_pred == y).mean()
    t = 0
    t0 = time.time()
    while True:
        x, y = sampler.sample()
        t, q_opt_state, loss_val = update(t, q_opt_state, x, y)

        t1 = time.time()
        if t1 - t0 > 1.0:
            print(
                t,
                "{:.2f}".format(loss_val),
                "{:.2f}".format(accuracy(q_opt_state, test_sampler) * 100))
            t0 = t1

if __name__ == "__main__":
    main()