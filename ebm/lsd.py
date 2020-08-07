import jax
import jax.numpy as jnp
from model.maker.model_maker import net_maker

class LSD_Learner:

    @staticmethod
    def __calc_sq_batch(q_params, x_batch, arg_q_apply_fun):
        def logq_sum(q_params, x_batch):
            logq_batch = arg_q_apply_fun(q_params, x_batch)
            return logq_batch.sum()
        sq_batch = jax.grad(logq_sum, argnums = 1)(q_params, x_batch) # ▽x(Log(q))
        return sq_batch

    @staticmethod
    def __calc_efficient_trace(f_params, x_batch, arg_f_apply_fun, rng):
        eps = jax.random.normal(rng, x_batch.shape)
        fx, vjp_fun = jax.vjp(arg_f_apply_fun, f_params, x_batch)
        dp, dx = vjp_fun(eps)
        return (dx * eps).sum(axis = -1), fx

    @staticmethod
    def __calc_exact_trace(f_params, x_batch, arg_f_apply_fun):
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
        tr_dfdx_batch, fx_batch = LSD_Learner.__calc_efficient_trace(f_params, x_batch, arg_f_apply_fun, rng)
        sq_batch = LSD_Learner.__calc_sq_batch(q_params, x_batch, arg_q_apply_fun)
        sq_fx_batch = (sq_batch * fx_batch).sum(axis = -1)
        lsd = (sq_fx_batch + tr_dfdx_batch).mean()
        f_norm = (fx_batch * fx_batch).sum(axis = -1).mean()
        return jnp.abs(lsd), f_norm

    @staticmethod
    def f_loss( q_params, f_params, x_batch, l2_weight,
                arg_q_apply_fun, arg_f_apply_fun, rng):
        lsd, f_norm =  LSD_Learner.calc_loss_metrics(   q_params, f_params, x_batch,
                                            arg_q_apply_fun, arg_f_apply_fun, rng)
        return -lsd + l2_weight * f_norm
        
    @staticmethod
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

    @staticmethod
    def get_scale(sampler, sample_num, x_dim):
        x = jnp.empty((0, x_dim))
        while x.shape[0] <= sample_num:
            x = jnp.append(x, sampler.sample(), axis = 0)
        return x.mean(), x.std()

