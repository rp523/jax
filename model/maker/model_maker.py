#coding: utf-8
import jax.random as jrandom

class net_maker():
    
    def __init__(self):
        self.__names = []       # each layer's name
        self.__input_names = [] # each layer's input layer name
        # layer process substance
        self.__init_funs = []   # initialize-functions
        self.__apply_funs = []  # apply-functions

    @staticmethod
    def merge_models(model_list):
        merged_model = net_maker()
        for part_model in model_list:
            merged_model.add_model(part_model)
        return merged_model

    def add_model(self, arg_model):
        names, input_names, init_funs, apply_funs = arg_model.get_layer_info()
        self.__names += names
        self.__input_names += input_names
        self.__init_funs += init_funs
        self.__apply_funs += apply_funs

    def get_layer_info(self):
        return  self.__names, \
                self.__input_names, \
                self.__init_funs, \
                self.__apply_funs
    
    def add_layer(  self,
                    layer,
                    name = None,
                    input_name = None):
        if name is not None:
            assert(not name in self.__names)
        self.__names.append(name)
        self.__input_names.append(input_name)

        init_fun, apply_fun = layer
        self.__init_funs.append(init_fun)
        self.__apply_funs.append(apply_fun)
    
    def __get_input_index(self, layer_index):
        assert(layer_index > 0)
        input_name = self.__input_names[layer_index]
        if input_name is None:
            return layer_index - 1
        else:
            assert(input_name in self.__names)
            return self.__names.index(input_name)

    @staticmethod    
    def weight_decay(params):
        ret = 0.0
        if isinstance(params, list) or isinstance(params, tuple):
            params = list(params)
            for param in params:
                ret += net_maker.weight_decay(param)
        else:
            ret += (params ** 2).sum()
        return ret

    def get_jax_model(self):
        n_layers = len(self.__init_funs)

        # initialize-function of whole model
        def init_fun(rng, input_shape):
            params = []
            output_shapes = []
            
            for idx, init_fun in enumerate(self.__init_funs):
                rng, layer_rng = jrandom.split(rng)
                if idx > 0:
                    input_shape = output_shapes[self.__get_input_index(idx)]
                    assert input_shape is not None
                output_shape, param = init_fun(layer_rng, input_shape)

                params.append(param)
                output_shapes.append(output_shape)

            return output_shape, params

        # apply-function of whole model
        def apply_fun(params, inputs, **kwargs):
            final_output = {}

            rng = kwargs.pop('rng', None)
            rngs = jrandom.split(rng, n_layers) if rng is not None else (None,) * n_layers
            for idx, (param, apply_fun, name, input_name, rng) in enumerate(zip(params, self.__apply_funs, self.__names, self.__input_names, rngs)):
                if idx > 0:
                    if input_name is None:
                        inputs = output
                    else:
                        inputs = final_output[input_name]
                output = apply_fun(param, inputs, rng=rng, **kwargs)    # apply
                if name is not None:
                    final_output[name] = output
            return final_output
        # return 2 functions
        return init_fun, apply_fun

if __name__ == "__main__":
    pass


