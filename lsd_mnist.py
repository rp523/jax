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
from sklearn.metrics import confusion_matrix
import seaborn as sns

SEED = 0
BATCH_SIZE = 128
X_DIM = 28 * 28
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
                    
def get_scale(sampler, sample_num, x_dim):
    x = jnp.empty((0, x_dim))
    while x.shape[0] <= sample_num:
        x_new, _ = sampler.sample()
        x = jnp.append(x, x_new.reshape((x_new.shape[0], -1)), axis = 0)
    return x.mean(axis = 0), x.std(axis = 0)

def jem(base_net, init_mu, init_sigma):
    base_init_fun, base_apply_fun = base_net
    def init_fun(rng, input_shape):
        rng_base, rng_mu, rng_sigma = jax.random.split(rng, 3)
        _, base_params = base_init_fun(rng_base, input_shape)
        designed_shape = tuple(input_shape[1:])
        assert(designed_shape == init_mu.shape)
        assert(designed_shape == init_sigma.shape)
        init_log_sigma = jnp.log(init_sigma)
        params = list(base_params) + [(init_mu, init_log_sigma)]
        return None, params
    def apply_fun(params, inputs, **kwargs):
        base_params = params[:-1]
        log_q_xy = base_apply_fun(base_params, inputs)
        log_q_x_base = jax.scipy.special.logsumexp(log_q_xy, axis = 1).reshape((log_q_xy.shape[0], 1))

        mu, log_sigma = params[-1]
        mu = mu.reshape(tuple([1] + list(mu.shape)))
        sigma = jnp.exp(log_sigma)
        log_gauss = - ((inputs - mu) ** 2) / (2 * sigma ** 2)
        log_gauss = log_gauss.sum(axis = -1)
        log_gauss = log_gauss.reshape((log_q_x_base.shape[0], -1))
        log_q_x = log_q_x_base + log_gauss
        '''
        mu, log_sigma = params[-1]
        mu = mu.reshape(tuple([1] + list(mu.shape)))
        sigma = jnp.exp(log_sigma)
        log_gauss = - ((inputs - mu) ** 2).sum(axis = -1) / (2 * sigma ** 2)
        log_gauss = log_gauss.reshape((inputs.shape[0], -1))

        base_params = params[:-1]
        log_q_xy = base_apply_fun(base_params, inputs) + log_gauss
        log_q_x_base = jax.scipy.special.logsumexp(log_q_xy, axis = 1).reshape((log_q_xy.shape[0], 1))
        log_q_x = log_q_x_base
        '''
        return {"class_logit" : log_q_xy,
                "class_prob" : jax.nn.softmax(log_q_xy),
                "log_density" : log_q_x}
    return init_fun, apply_fun

def make_conf_mat(q_opt_state, arg_q_get_params, arg_q_apply_fun_raw, arg_test_sampler):
    q_params = arg_q_get_params(q_opt_state)
    x, y_correct = arg_test_sampler.sample(get_all = True)
    result = arg_q_apply_fun_raw(q_params, x)
    y_pred = result["class_prob"].argmax(axis = -1)
    # calc confusion matrix
    cm = confusion_matrix(y_correct, y_pred, normalize = "true")

    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, ylabel = "annotated", xlabel = "predicted")
    sns.heatmap(cm, annot = True, cmap = "jet", fmt="1.4f", ax = ax,
                square = True)
    plt.savefig('conf_mat.png')

def show_sample(class_num, q_opt_state, arg_q_get_params, arg_q_apply_fun_raw):
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
            sc = 1
            def metric_func(x, q_params, idx):
                ret = arg_q_apply_fun_raw(q_params, x)["log_density"]
                #ret = arg_q_apply_fun_raw(q_params, x)["class_logit"].flatten()[idx]
                assert(ret.size == 1)
                return ret.sum()
            met = 9999999
            record_num = 100
            met_record = np.ones(record_num) * met
            cnt = 0
            t0 = time.time()
            while True:
                dfdx = jax.grad(metric_func)(x, q_params, h)
                rng, rng1 = jax.random.split(rng)
                x += 1E-0 * dfdx + 1E-2 * jax.random.uniform(rng1, x.shape)
                met = metric_func(x, q_params, h)
                met_record[cnt] = float(met)
                cnt = (cnt + 1) % record_num
                sc += 1
                t1 = time.time()
                if t1 - t0 > 10:
                    print(
                        h,
                        w,
                        sc,
                        "{:.3f}".format(met),
                        "{:.3f}".format(met_record.max() - met_record.min()))
                    t0 = t1
                if (met_record.max() - met_record.min()) < 1E-2:
                    break
            x = Mnist.quantize(x.reshape((28, 28)))
            y = np.asarray(x).astype(np.uint8)
            all_arr[h * 28: (h + 1) * 28, w * 28 : (w + 1) * 28] = y
    pil = Image.fromarray(all_arr)
    w, h = pil.size
    pil = pil.resize((2*w, 2*h))
    pil.show()

def show_graph():
    plt.clf()
    plot_points = 50

    colors = ["blue", "blue", "red", "red", "purple"]
    linestyles = ["solid", "dashed", "solid", "dashed", "dashed"]
    titles = [  "softmax",
                "logit",
                "尤度モデルsoftmax",
                "尤度モデルlogit",
                "尤度モデルdensity",
                ]
    keys = ["class_prob",
            "class_logit",
            "class_prob",
            "class_logit",
            "log_density",
            ]
    for j, model_class_num in enumerate([9]):
        if j == 0:
            #paths = ["exp/05_class_allclass"] * 2 + ["exp/03_joint_allclasses"] * 3
            paths = ["exp/05_class_non0"] * 2 + ["exp/04_joint_non0"] * 3
            paths = ["exp/08_class_non0train"] * 2 + ["exp/07_joint_non0train"] * 3
        rng = jax.random.PRNGKey(0)
        test_sampler = Mnist(rng, 10000, "test", one_hot = False, dequantize = True, flatten = True)
            
        for i, (xlabel, key, weight_path) in enumerate(zip(titles, keys, paths)):
            dummy = 0.0
            _, q_apply_fun_raw = jem(mlp(model_class_num), dummy, dummy)
            q_params, _ = pickle.load(open(os.path.join(weight_path, "params.bin"), "rb"))
            x, y = test_sampler.sample(get_all = True)
            pred = q_apply_fun_raw(q_params, x)
            metrics = pred[key]
            if key in ["class_prob", "class_logit"]:
                metrics = metrics.max(axis = -1)
            elif key in ["log_density"]:
                metrics = metrics.flatten()
            else:
                assert(0)
            met_vec = np.linspace(metrics.min(), metrics.max(), plot_points)
            plot_x = np.empty(0)
            plot_y0 = np.empty(0)
            plot_y1 = np.empty(0)
            for k in range(len(met_vec)):
                if k < met_vec.size - 1:
                    print(j, i, k, met_vec[k])
                    '''
                    if j == 0:
                        is_valid = np.logical_and(met_vec[k] <= metrics, metrics < met_vec[k + 1])
                        if is_valid.any():
                            y_ans = y[is_valid]
                            y_prd = pred["class_prob"].argmax(axis = -1)[is_valid]
                            ok_rate = (y_ans == y_prd).mean()
                            plot_y0 = np.append(plot_y0, ok_rate)
                            plot_y1 = np.append(plot_y1, is_valid.mean())
                            plot_x = np.append(plot_x, 0.5 * (met_vec[k] + met_vec[k + 1]))#cum + is_valid.mean())
                        is_valid = met_vec[k] <= metrics
                        if is_valid.any():
                            y_ans = y[is_valid]
                            y_prd = pred["class_prob"].argmax(axis = -1)[is_valid]
                            ok_rate = (y_ans == y_prd).mean()
                            plot_y = np.append(plot_y, ok_rate)
                            plot_x = np.append(plot_x, is_valid.mean())#cum + is_valid.mean())
                    '''
                    if j == 0:
                        #is_valid = np.logical_and(met_vec[k] <= metrics, metrics < met_vec[k + 1]) 
                        is_valid = (met_vec[k] <= metrics)
                        all_y_prd = pred["class_prob"].argmax(axis = -1) + 1
                        if is_valid.any():
                            y_prd = all_y_prd[is_valid]
                            y_ans = y[is_valid]
                            correct_rate = (y_prd == y_ans).sum() / (y != 0).sum()
                            unseen_rate = (y_ans == 0).sum() / (y == 0).sum()
                        else:
                            correct_rate = 0.0
                            unseen_rate = 1.0

                        plot_x  = np.append(plot_x,  1.0 - unseen_rate)
                        plot_y0 = np.append(plot_y0, correct_rate)
            #ax1 = fig.add_subplot(3, 5, j * 5 + i + 1)
            #ax1 = fig.add_subplot(111)
            plt.plot(plot_x, plot_y0, label = xlabel, color = colors[i], linestyle = linestyles[i])
            #plt.plot(plot_x, plot_y1, color = "red")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    FONT = "Myrica M"
    plt.xlabel("未学習画像のうち、未学習と判定できた割合", fontname = FONT)
    plt.ylabel("学習済画像のうち、学習済と判定＆正解した割合", fontname = FONT)
    plt.legend(prop={"family":FONT})
    plt.title("未学習判定と正解率のトレードオフ", fontname = FONT)
    plt.savefig("unk_graphs.png")
    plt.show()

def main(is_eval, class_num):
    _rng = jax.random.PRNGKey(SEED)
    
    _rng, rng_d, rng_q, rng_f = jax.random.split(_rng, 4)
    train_sampler = Mnist(rng_d, BATCH_SIZE, "train", one_hot = True, dequantize = True, flatten = True, class_num = class_num)#, remove_classes=[0])
    test_sampler = Mnist(rng_d, 10000, "test", one_hot = False, dequantize = True, flatten = True, class_num = class_num)#, remove_classes=[0])
    mu, sigma = get_scale(train_sampler, 1000, X_DIM)
    #print("mu={}, sigma={}".format(mu, sigma))

    q_init_fun, q_apply_fun_raw = jem(mlp(class_num), mu, sigma)
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
        pred_idx = jnp.argmax(pred_prob, axis = -1)# + 1
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
                    "qx_loss={:.5f}".format(q_loss_val / (t - t_old)),
                    "fx_loss={:.5f}".format(-1 * f_loss_val / (c - c_old)),
                    ) 
            q_loss_val, f_loss_val = 0.0, 0.0
            t_old, c_old =  t, c
            t0 = t1
            pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(SAVE_PATH, "wb"))

    show_sample(class_num, q_opt_state, q_get_params, q_apply_fun_raw)
    return
    make_conf_mat(q_opt_state, q_get_params, q_apply_fun_raw, test_sampler)

if __name__ == "__main__":
    CLASS_NUM = 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action = "store_true")
    arg = parser.parse_args()
    #arg.eval = True
    #show_graph()
    main(arg.eval, CLASS_NUM)
    print("Done.")
