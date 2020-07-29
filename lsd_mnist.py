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
from dataset.fashion_mnist import FashionMnist
from ebm.lsd import LSD_Learner
from model.maker.model_maker import net_maker
from sklearn.metrics import confusion_matrix
import seaborn as sns
import hydra

X_DIM = 28 * 28

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
        '''
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
        log_gauss = - ((inputs - mu) ** 2) / (2 * sigma ** 2)
        log_gauss = log_gauss.sum(axis = -1)
        log_gauss = log_gauss.reshape((inputs.shape[0], -1))

        base_params = params[:-1]
        log_q_xy_raw = base_apply_fun(base_params, inputs)
        log_q_xy = log_q_xy_raw + log_gauss
        log_q_x_raw = jax.scipy.special.logsumexp(log_q_xy, axis = 1).reshape((log_q_xy_raw.shape[0], 1))
        log_q_x = jax.scipy.special.logsumexp(log_q_xy, axis = 1).reshape((log_q_xy.shape[0], 1))
        return {"class_logit_raw" : log_q_xy_raw,
                "class_logit" : log_q_xy,
                "class_prob" : jax.nn.softmax(log_q_xy),
                "log_density_raw" : log_q_x_raw,
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
    
    sample_h = class_num 
    sample_w = 3
    all_arr = np.zeros((sample_h * 28, sample_w * 28), dtype = np.uint8)
    rng = jax.random.PRNGKey(1)
    for h in range(sample_h):
        for w in range(sample_w):
            rng, rng1 = jax.random.split(rng)
            x = jax.random.normal(rng1, (1, dim)) * sigma + mu
            sc = 1
            def metric_func(x, q_params, idx):
                #ret = arg_q_apply_fun_raw(q_params, x)["log_density"]
                #assert(arg_q_apply_fun_raw(q_params, x)["class_logit"].size == class_num)
                ret = arg_q_apply_fun_raw(q_params, x)["class_logit"].flatten()[idx]
                assert(ret.size == 1)
                return ret.sum()
            met = 9999999
            record_num = 100
            met_record = np.ones(record_num) * met
            cnt = 0
            t0 = time.time()
            for lr in np.linspace(1E-0, 1E-5, 999999):
                dfdx = jax.grad(metric_func)(x, q_params, h)
                rng, rng1 = jax.random.split(rng)
                epsilon = jax.random.uniform(rng1, x.shape)
                x += 1.0 * dfdx + lr * epsilon
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
            x = Mnist.quantize(x.reshape((28, 28)))
            y = np.asarray(x).astype(np.uint8)
            all_arr[h * 28: (h + 1) * 28, w * 28 : (w + 1) * 28] = y
    pil = Image.fromarray(all_arr)
    w, h = pil.size
    pil = pil.resize((2*w, 2*h))
    pil.save("sampled.png")
    pil.show()

def fashion_test():
    plt.clf()
    plot_points = 200

    titles = [  "通常モデルのsoftmax最大値",
                "通常モデルのlogit最大値",
                "複合モデルの周辺尤度",
                "複合モデルのsoftmax最大値",
                "複合モデルのlogit最大値",
                ]
    keys = ["class_prob",
            "class_logit_raw",
            "log_density",
            "class_prob",
            "class_logit",
            ]
    #paths = ["exp/05_class_allclass"] * 2 + ["exp/03_joint_allclasses"] * 3
    weight_paths = ["/home/isgsktyktt/work/multirun/2020-07-29/08-45-26/0", # classify
                    "/home/isgsktyktt/work/multirun/2020-07-29/08-45-26/0", # classify
                    "/home/isgsktyktt/work/multirun/2020-07-28/21-59-23/5", # joint
                    "/home/isgsktyktt/work/multirun/2020-07-28/21-59-23/5", # joint
                    "/home/isgsktyktt/work/multirun/2020-07-28/21-59-23/5", # joint
            ]
    rng = jax.random.PRNGKey(0)
    mnist = Mnist(rng, 10000, "test", one_hot = False, dequantize = True, flatten = True, dir_path = ".")
    fashion = FashionMnist(rng, 10000, "test", one_hot = False, dequantize = True, flatten = True, dir_path = ".")
        
    for i, (xlabel, key, weight_path) in enumerate(zip(titles, keys, weight_paths)):
        dummy = 0.0
        _, q_apply_fun_raw = jem(mlp(10), dummy, dummy)
        q_params, _ = pickle.load(open(os.path.join(weight_path, "params.bin"), "rb"))
        mx, my = mnist.sample(get_all = True)
        fx, fy = fashion.sample(get_all = True)
        my_pred_dict = q_apply_fun_raw(q_params, mx)
        fy_pred_dict = q_apply_fun_raw(q_params, fx)
        my_metrics = my_pred_dict[key].max(axis = -1)
        fy_metrics = fy_pred_dict[key].max(axis = -1)

        metrics = np.sort(np.append(my_metrics, fy_metrics))
        met_idx = (np.linspace(0.0, 1.0, plot_points + 1)[:-1] * metrics.size).astype(np.int32)
        met_vec = metrics[met_idx]
        assert(met_vec.size == plot_points)
        plot_x = np.zeros(plot_points)
        plot_y = np.zeros(plot_points)
        for k, met_val in enumerate(met_vec):
            m_valid = (met_val <= my_metrics)
            f_valid = (met_val >  fy_metrics)
            all_my_pred = my_pred_dict["class_prob"].argmax(axis = -1)
            #all_fy_pred = fy_pred_dict["class_prob"].argmax(axis = -1)
            # y: correct rate of learned dataset
            y_val = 0.0
            if m_valid.any():
                y_prd = all_my_pred[m_valid]
                y_ans = my[m_valid]
                correct_rate = (y_prd == y_ans).sum() / my.size
                y_val = correct_rate
            # x: uncertainty detect rate of unlearned dataset
            uncertain_num = f_valid.sum()
            uncertain_rate = uncertain_num / fy.size
            x_val = uncertain_rate

            plot_x[k] = x_val
            plot_y[k] = y_val
            print(i, k, met_val, x_val, y_val)

        plt.plot(plot_x, plot_y, label = xlabel)
        if 0:#i >= 2:
            break

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    FONT = "Myrica M"
    plt.xlabel("未学習画像のうち、未学習と判定できた割合", fontname = FONT)
    plt.ylabel("学習済画像のうち、学習済と判定＆正解できた割合", fontname = FONT)
    plt.legend(prop={"family":FONT})
    plt.title("未学習判定と正解率のトレードオフ", fontname = FONT)
    plt.savefig("unk_graphs_.png")
    plt.show()

@hydra.main(config_path="lsd_mnist.yaml")
def main(cfg):
    print(cfg.pretty())

    _rng = jax.random.PRNGKey(cfg.seed)
    _rng, rng_d, rng_q, rng_f = jax.random.split(_rng, 4)

    train_sampler = Mnist(rng_d, cfg.optim.batch_size, "train", one_hot = True, dequantize = True, flatten = True, dir_path = hydra.utils.get_original_cwd(),
                            remove_classes = cfg.data.remove_class, remove_col_too = cfg.data.remove_col_too)
    test_sampler = Mnist(rng_d, 10000, "test", one_hot = False, dequantize = True, flatten = True, dir_path = hydra.utils.get_original_cwd(),
                            remove_classes = cfg.data.remove_class, remove_col_too = cfg.data.remove_col_too)
    mu, sigma = get_scale(train_sampler, 1000, X_DIM)

    train_class_num = 10 - len(cfg.data.remove_class) * int(cfg.data.remove_col_too)
    q_init_fun, q_apply_fun_raw = jem(mlp(  train_class_num),
                                            mu,
                                            sigma)
    f_init_fun, f_apply_fun = mlp(X_DIM)
    q_opt_init, q_opt_update, q_get_params = optimizers.adam(cfg.optim.q_lr)
    f_opt_init, f_opt_update, f_get_params = optimizers.adam(cfg.optim.f_lr)

    def q_apply_fun_classify(q_params, x_batch):
        return q_apply_fun_raw(q_params, x_batch)["class_prob"]
    def q_apply_fun_density(q_params, x_batch):
        return q_apply_fun_raw(q_params, x_batch)["log_density"]
    def classify_loss(q_params, x_batch, y_batch):
        y_pred = q_apply_fun_classify(q_params, x_batch)
        cross_entropy = (- y_batch * ((1.0 - y_pred) ** cfg.optim.focal_gamma) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        return cross_entropy
    def accuracy(q_params, arg_test_sampler):
        x_batch, y_batch = arg_test_sampler.sample(get_all = True)
        pred_prob = q_apply_fun_classify(q_params, x_batch)
        pred_idx = jnp.argmax(pred_prob, axis = -1)# + 1
        return (y_batch == pred_idx).mean()
    def q_loss(q_params, f_params, x_batch, y_batch, rng):
        loss = 0.0
        if cfg.optim.lsd_weight > 0.0:
            lsd, _ = LSD_Learner.calc_loss_metrics(q_params, f_params, x_batch, q_apply_fun_density, f_apply_fun, rng)
            loss += cfg.optim.lsd_weight * lsd
        loss += cfg.optim.weight_decay * net_maker.weight_decay(q_params)
        if cfg.optim.classify_weight > 0.0:
            loss += cfg.optim.classify_weight * classify_loss(q_params, x_batch, y_batch)
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
        if cfg.optim.lsd_weight > 0.0:
            loss_val, grad_val = jax.value_and_grad(LSD_Learner.f_loss, argnums = 1)(q_params, f_params, x_batch, cfg.optim.critic_l2, q_apply_fun_density, f_apply_fun, rng)
            f_opt_state = f_opt_update(c_cnt, grad_val, f_opt_state)
        return (c_cnt + 1), f_opt_state, loss_val
    if not os.path.exists(cfg.save_path):
        _, q_init_params = q_init_fun(rng_q, (cfg.optim.batch_size, X_DIM))
        _, f_init_params = f_init_fun(rng_f, (cfg.optim.batch_size, X_DIM))
    else:
        (q_init_params, f_init_params) = pickle.load(open(cfg.save_path, "rb"))
        print("LODADED INIT WEIGHT")
    q_opt_state = q_opt_init(q_init_params)
    f_opt_state = f_opt_init(f_init_params)

    t0 = time.time()
    t = c = 0
    t_old = c_old = 0
    q_loss_val, f_loss_val = 0.0, 0.0
    
    cnt = 0
    if cfg.optim.epoch_num > 0:
        while True:
            # learning
            x_batch, y_batch = train_sampler.sample()
            rng1, _rng = jax.random.split(_rng)
            if cnt % cfg.optim.critic_loop == 0:
                # joint model
                t, q_opt_state, q_loss_val1 = q_update(t, q_opt_state, f_opt_state, x_batch, y_batch, rng1)
                q_loss_val += q_loss_val1
            else:
                # cirtic
                c, f_opt_state, f_loss_val1 = f_update(c, q_opt_state, f_opt_state, x_batch, rng1)
                f_loss_val += f_loss_val1
            cnt += 1

            t1 = time.time()
            if ((t1 - t0 > 20.0) and (t > t_old) and (c > c_old)) \
                or (cnt / (60000 / cfg.optim.batch_size) > cfg.optim.epoch_num):
                log_txt = ""
                for txt in ["{:.2f}epoch".format(cnt / (60000 / cfg.optim.batch_size)),
                            "{:.2f}sec".format(t1 - t0),
                            "{:.2f}%".format(accuracy(q_get_params(q_opt_state), test_sampler) * 100),
                            "qx_loss={:.5f}".format(q_loss_val / (t - t_old)),
                            "fx_loss={:.5f}".format(-1 * f_loss_val / (c - c_old)),
                            ]:
                    if log_txt != "":
                        log_txt += ","
                    log_txt += txt
                print(log_txt)
                with open("log.txt", "a") as f:
                    f.write("{}\n".format(log_txt))

                q_loss_val, f_loss_val = 0.0, 0.0
                t_old, c_old =  t, c
                t0 = t1
                pickle.dump((q_get_params(q_opt_state), f_get_params(f_opt_state)), open(cfg.save_path, "wb"))
            
            if cnt / (60000 / cfg.optim.batch_size) > cfg.optim.epoch_num:
                break

    show_sample(train_class_num, q_opt_state, q_get_params, q_apply_fun_raw)
    return
    make_conf_mat(q_opt_state, q_get_params, q_apply_fun_raw, test_sampler)

if __name__ == "__main__":
    #fashion_test()
    main()
    print("Done.")
