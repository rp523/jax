import numpy as onp
import os, time

import jax
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random, value_and_grad, device_put, tree_map

class CheckPoint:
    def __recursive_proc(self, jax_params, infunc, dir, cnt = 0, recursive = 0):
        if isinstance(jax_params, list) or isinstance(jax_params, tuple):
            tmp = list(jax_params)
            for i in range(len(list(jax_params))):
                tmp[i], cnt = self.__recursive_proc(jax_params[i], infunc, dir, cnt, recursive + 1)
            if isinstance(jax_params, list):
                jax_params = list(tmp)
            elif isinstance(jax_params, tuple):
                jax_params = tuple(tmp)
        else:
            jax_params = infunc(jax_params, dir, cnt)
            cnt += 1
        
        if recursive == 0:
            return jax_params
        else:
            return jax_params, cnt

    def __load_func(self, val, dir, cnt):
        return onp.load(os.path.join(dir, "{}.npy".format(cnt)), allow_pickle = True)
    
    def __save_func(self, val, dir, cnt):
        onp.save(os.path.join(dir, "{}.npy".format(cnt)), val)
        return val
    def __equal_func(self, val, dir, cnt):
        loaded_arr = onp.load(os.path.join(dir, "{}.npy".format(cnt)), allow_pickle = True)
        assert((val == loaded_arr).all())
    def __not_equal_func(self, val, dir, cnt):
        loaded_arr = onp.load(os.path.join(dir, "{}.npy".format(cnt)), allow_pickle = True)
        assert(not (val == loaded_arr).all())
    
    def load_params(self, sample_params, dir):
        return self.__recursive_proc(sample_params,
                                     self.__load_func,
                                     dir)

    def save_params(self, params, dir):
        self.__recursive_proc(params,
                              self.__save_func,
                              dir)
