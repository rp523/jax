import os, time, pickle, argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, Dense, elementwise, FanOut, FanInSum, parallel, Identity, Conv, Tanh
import jax.experimental.optimizers as optimizers
from dataset.mnist import Mnist
from ebm.lsd import LSD_Learner
from model.maker.model_maker import net_maker

SEED = 0
BATCH_SIZE = 128
X_DIM = 28 * 28
CLASS_NUM = 10
Q_LR = 1E-3
F_LR = 1E-3
LAMBDA = 0.5
C = 10
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
        log_q_x_base = jax.scipy.special.logsumexp(log_q_xy, axis = 1).reshape((log_q_xy.shape[0], 1))

        mu, log_sigma = params[-1]
        mu = mu.reshape(tuple([1] + list(mu.shape)))
        sigma = jnp.exp(log_sigma)
        log_gauss = - ((inputs - mu) ** 2).sum(axis = -1) / (2 * sigma ** 2)
        log_gauss = log_gauss.reshape((log_q_x_base.shape[0], -1))
        log_q_x = log_q_x_base + log_gauss
        return {"class_logit" : log_q_xy,
                "class_prob" : jax.nn.softmax(log_q_xy),
                "log_density" : log_q_x}
    return init_fun, apply_fun

def show_result(q_opt_state, arg_q_get_params, arg_q_apply_fun_raw, arg_test_sampler):
    q_params = arg_q_get_params(q_opt_state)

    # sample data
    mu, log_sigma = q_params[-1]
    sigma = jnp.exp(log_sigma)
    dim = mu.size
    
    sample_h = 1
    sample_w = 3
    all_arr = np.zeros((sample_h * 28, sample_w * 28), dtype = np.uint8)
    rng = jax.random.PRNGKey(1)
    for h in range(sample_h):
        for w in range(sample_w):
            rng, rng1 = jax.random.split(rng)
            x = jax.random.normal(rng1, (1, dim)) * sigma + mu
            def metric_func(x, q_params):
                return arg_q_apply_fun_raw(q_params, x)["log_density"].sum()
            met = 9999999
            record_num = 100
            met_record = np.ones(record_num) * met
            cnt = 0
            while True:
                dfdx = jax.grad(metric_func)(x, q_params)
                rng, rng1 = jax.random.split(rng)
                x += 1E-0 * dfdx + 1E-2 * jax.random.uniform(rng1, x.shape)
                met = metric_func(x, q_params)
                met_record[cnt] = float(met)
                cnt = (cnt + 1) % record_num
                print(h, w, met_record.max() - met_record.min())
                if (met_record.max() - met_record.min()) < 1E-2:
                    break
            x = Mnist.quantize(x.reshape((28, 28)))
            y = np.asarray(x).astype(np.uint8)
            all_arr[h * 28: (h + 1) * 28, w * 28 : (w + 1) * 28] = y
    pil = Image.fromarray(all_arr)
    pil.show()
    return
    '''
    q = np.ones((sample_num, dim)) * (-999999)
    not_update_cnt = 0
    while not_update_cnt < 10000:
        rng, rng1 = jax.random.split(rng)
        x1 = np.asarray(jax.random.uniform(rng1, (sample_num, dim)) * sigma + mu)
        q1 = np.asarray(arg_q_apply_fun_raw(q_params, x1)["log_density"])
        q1 = q1.flatten()
        next_idx = np.argsort(q1)
        x1 = x1[next_idx]
        q = q[next_idx]

        q_min = q.min()
        update_indexes = q1 > q_min
        if update_indexes.any():
            updates = q1[update_indexes]
            q = np.append(updates, q[updates.size:])
            x = np.append(x1[:updates.size], x[updates.size:], axis = 0)
            q = np.asarray(arg_q_apply_fun_raw(q_params, x)["log_density"]).flatten()
            next_idx = np.argsort(q)
            x = x[next_idx]
            q = q[next_idx]
            not_update_cnt = 0
        else:
            not_update_cnt += 1
        print(q.min(), q.max())
    '''

    # visualize classify graph
    test_x, test_lbls = arg_test_sampler.sample(get_all = True)
    result_dict = arg_q_apply_fun_raw(q_params, test_x)
    pred_lbls = result_dict["class_prob"].argmax(axis = -1)
    pred_prob = result_dict["class_prob"].max(axis = -1)
    pred_logi = result_dict["class_logit"].max(axis = -1)
    pred_dens = result_dict["log_density"].flatten()

    correct = (pred_lbls == test_lbls)
    incorrect = (pred_lbls != test_lbls)

    plt.clf()
    fig = plt.figure(figsize=(15, 5))
    plot_points = 50

    ax = fig.add_subplot(141)
    plot_x = np.linspace(0,1,2)
    ones = np.ones(plot_x.shape)
    plt.fill_between(plot_x, 0.0 * ones, 0.1 * ones, facecolor = 'g', alpha=0.5)
    plt.fill_between(plot_x, 0.1 * ones, ones      , facecolor = 'b', alpha=0.5)
    plt.xlim(plot_x.min(), plot_x.max())
    plt.title("ideal")

    for i, (title, metrics, pos) in enumerate( [["softmax_score", pred_prob, 142],
                                                ["logit_score", pred_logi, 143],
                                                ["log_density", pred_dens, 144]]):
        ax = fig.add_subplot(pos)
        accuracy = np.zeros((plot_points,), dtype = np.float32)
        error = np.zeros((plot_points,), dtype = np.float32)
        unlearn = np.zeros((plot_points,), dtype = np.float32)
        plot_metrics = np.zeros((plot_points,), dtype = np.float32)
        for i, metric in enumerate(tqdm(np.linspace(metrics.min(), metrics.max(), plot_points))):
            is_output = (float(metric) <= metrics)
            accuracy[i] = float(correct[is_output].sum() / is_output.size)
            error[i] = float(incorrect[is_output].sum() / is_output.size)
            unlearn[i] = 1.0 - accuracy[i] - error[i] 
            plot_metrics[i] = float(metric)
        
        zeros = np.zeros(error.shape)
        ones = np.ones(error.shape)
        plt.fill_between(plot_metrics, zeros, error, facecolor = 'r', alpha=0.5)
        plt.fill_between(plot_metrics, error, error + unlearn, facecolor = 'g', alpha=0.5)
        plt.fill_between(plot_metrics, error + unlearn, ones, facecolor = 'b', alpha=0.5)
        plt.xlim(plot_metrics.min(), plot_metrics.max())
        plt.title(title)
        if i == 0:
            plt.legend()
    plt.savefig("unk_graphs.png")
    plt.show()

def get_scale(sampler, sample_num, x_dim):
    x = jnp.empty((0, x_dim))
    while x.shape[0] <= sample_num:
        x_new, _ = sampler.sample()
        x = jnp.append(x, x_new.reshape((x_new.shape[0], -1)), axis = 0)
    return x.mean(), x.std()

def main(is_eval):
    _rng = jax.random.PRNGKey(SEED)
    
    _rng, rng_d, rng_q, rng_f = jax.random.split(_rng, 4)
    train_sampler = Mnist(rng_d, BATCH_SIZE, "train", one_hot = True, dequantize = True, flatten = True)#, remove_classes=[0])
    test_sampler = Mnist(rng_d, 10000, "test", one_hot = False, dequantize = True, flatten = True)
    mu, sigma = get_scale(train_sampler, 1000, X_DIM)
    print("mu={}, sigma={}".format(mu, sigma))

    q_init_fun, q_apply_fun_raw = jem(mlp(CLASS_NUM), mu, sigma)
    f_init_fun, f_apply_fun = mlp(X_DIM)
    q_opt_init, q_opt_update, q_get_params = optimizers.adam(Q_LR)
    f_opt_init, f_opt_update, f_get_params = optimizers.adam(F_LR)

    def q_apply_fun_classify(q_params, x_batch):
        return q_apply_fun_raw(q_params, x_batch)["class_prob"]
    def q_apply_fun_density(q_params, x_batch):
        return q_apply_fun_raw(q_params, x_batch)["log_density"]
    def classify_loss(q_params, x_batch, y_batch):
        y_pred = q_apply_fun_classify(q_params, x_batch)
        cross_entropy = (- y_batch * ((1.0 - y_pred) ** FOCAL_GAMMA) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        return cross_entropy
    def accuracy(q_params, arg_test_sampler):
        x_batch, y_batch = arg_test_sampler.sample(get_all = True)
        pred_prob = q_apply_fun_classify(q_params, x_batch)
        pred_idx = jnp.argmax(pred_prob, axis = -1)
        return (y_batch == pred_idx).mean()
    def q_loss(q_params, f_params, x_batch, y_batch, rng):
        loss = 0.0
        lsd, _ = LSD_Learner.calc_loss_metrics(q_params, f_params, x_batch, q_apply_fun_density, f_apply_fun, rng)
        loss += lsd
        loss += 1E-5 * net_maker.weight_decay(q_params)
        loss += classify_loss(q_params, x_batch, y_batch)
        return loss
    @jax.jit
    def q_update(t_cnt, q_opt_state, f_opt_state, x_batch, y_batch, rng):
        q_params = q_get_params(q_opt_state)
        f_params = f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(q_loss)(q_params, f_params, x_batch, y_batch, rng)
        q_opt_state = q_opt_update(t_cnt, grad_val, q_opt_state)
        return (t_cnt + 1), q_opt_state, loss_val
    @jax.jit
    def f_update(c_cnt, q_opt_state, f_opt_state, x_batch, rng):
        loss_val = 0.0
        q_params = q_get_params(q_opt_state)
        f_params = f_get_params(f_opt_state)
        loss_val, grad_val = jax.value_and_grad(LSD_Learner.f_loss, argnums = 1)(q_params, f_params, x_batch, LAMBDA, q_apply_fun_density, f_apply_fun, rng)
        f_opt_state = f_opt_update(c_cnt, grad_val, f_opt_state)
        return (c_cnt + 1), f_opt_state, loss_val
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
    t = c = 0
    t_old = c_old = 0
    q_loss_val, f_loss_val = 0.0, 0.0
    
    cnt = 0
    while not is_eval:
        # learning
        x_batch, y_batch = train_sampler.sample()
        rng1, _rng = jax.random.split(_rng)
        if cnt % C == 0:
            # joint model
            t, q_opt_state, q_loss_val1 = q_update(t, q_opt_state, f_opt_state, x_batch, y_batch, rng1)
            q_loss_val += q_loss_val1
        else:
            # cirtic
            c, f_opt_state, f_loss_val1 = f_update(c, q_opt_state, f_opt_state, x_batch, rng1)
            f_loss_val += f_loss_val1
        cnt += 1

        t1 = time.time()
        if t1 - t0 > 20.0:
            print(  "{:.2f}epoch".format(cnt / (60000 / BATCH_SIZE)),
                    "{:.2f}sec".format(t1 - t0),
                    "{:.2f}%".format(accuracy(q_get_params(q_opt_state), test_sampler) * 100),
                    "qx_loss={:.2f}".format(q_loss_val / (t - t_old)),
                    "fx_loss={:.2f}".format(-1 * f_loss_val / (c - c_old)),
                    ) 
            q_loss_val, f_loss_val = 0.0, 0.0
            t_old, c_old =  t, c
            t0 = t1
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))

    show_result(q_opt_state, q_get_params, q_apply_fun_raw, test_sampler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action = "store_true")
    arg = parser.parse_args()
    #arg.eval = True
    main(arg.eval)
    print("Done.")
