#coding: utf-8
import jax.random as jrandom

class net_maker():
    
    def __init__(self):
        self.__init_funs = []
        self.__apply_funs = []
        self.__names = []
        self.__input_names = []
        self.__is_output = []
    
    def add_layer(self, layer,
                        name = None,
                        input_name = None,
                        is_output = False):
        init_fun, apply_fun = layer
        self.__init_funs.append(init_fun)
        self.__apply_funs.append(apply_fun)
        if name != name:
            assert(not name in self.__names)
        self.__names.append(name)
        self.__input_names.append(input_name)
        self.__is_output.append(is_output)
    
    def make_jax_model(self):
        n_layers = len(self.__init_funs)
        def init_fun(rng, input_shape):
            params = []
            for init_fun in self.__init_funs:
                rng, layer_rng = jrandom.split(rng)
                input_shape, param = init_fun(layer_rng, input_shape)
                params.append(param)
            return input_shape, params
        def apply_fun(params, inputs, **kwargs):
            input_set_dict = {}
            output_dict = {}
            rng = kwargs.pop('rng', None)
            rngs = jrandom.split(rng, n_layers) if rng is not None else (None,) * n_layers
            output = None
            for l, (fun, param, name, input_name, is_output, rng) in enumerate(zip(self.__apply_funs,
                                                                            params,
                                                                            self.__names,
                                                                            self.__input_names,
                                                                            self.__is_output,
                                                                            rngs)):
                if l == 0:
                    layer_inputs = inputs
                else:
                    if input_name != None: # input layer is specified
                        assert(input_name in self.__names)
                        layer_inputs = input_set_dict[input_name]
                    else:
                        assert(output != None)
                        layer_inputs = output   # previous result
                output = fun(param, layer_inputs, rng=rng, **kwargs)    # apply
                # save result
                if name != None:
                    input_set_dict[name] = output   # for future input use
                if is_output:
                    assert(name != None)
                    output_dict[name] = output  # for output
            return output_dict
        return init_fun, apply_fun