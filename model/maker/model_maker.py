#coding: utf-8
import jax.random as jrandom

class net_maker():
    
    def __init__(self, prev_model = None):
        self.__names = []       # each layer's name
        self.__input_names = [] # each layer's input layer name
        # layer process substance
        self.__init_funs = []   # initialize-functions
        self.__apply_funs = []  # apply-functions

        if prev_model is not None:  # given previously-defined model
            self.__names, self.__input_names, self.__init_funs, self.__apply_funs = prev_model.get_layer_info()
    
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
    
    def get_jax_model(self):
        n_layers = len(self.__init_funs)
        input_shapes = [input]

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
            for idx, (param, apply_fun, name, rng) in enumerate(zip(params, self.__apply_funs, self.__names, rngs)):
                if idx > 0:
                    input_name = self.__input_names[idx]
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


