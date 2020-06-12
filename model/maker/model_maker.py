#coding: utf-8
import jax.random as jrandom

class net_maker():
    
    def __init__(self, prev_model = None):
        self.__names = []       # each layer's name
        self.__input_names = [] # each layer's input layer name
        # layer process substance
        self.__init_funs = []   # initialize-functions
        self.__apply_funs = []  # apply-functions

        if prev_model != None:  # when conncet to previously-defined model
            # fit funs for this class
            init_fun, apply_fun = prev_model.get_jax_model()
            init_funs, apply_funs = [init_fun], [apply_fun]
            # make dummy arrays
            names, input_names, is_output = [None], [None], [False]
            if isinstance(prev_model, net_maker):
                # previous model is instanced by this class
                # able to get detailed array info
                names, input_names, is_output, init_funs, apply_funs = prev_model.get_layer_info()
            self.__names = names
            self.__input_names = input_names
            self.__init_funs = init_funs
            self.__apply_funs = apply_funs
    
    def get_layer_info(self):
        return  self.__names, \
                self.__input_names, \
                self.__init_funs, \
                self.__apply_funs
    
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
    
    def get_jax_model(self):
        n_layers = len(self.__init_funs)

        # initialize-function of whole model
        def init_fun(rng, input_shape):
            params = []
            for init_fun in self.__init_funs:
                rng, layer_rng = jrandom.split(rng)
                output_shape, param = init_fun(layer_rng, input_shape)
                params.append(param)
            return output_shape, params

        # apply-function of whole model
        def apply_fun(params, inputs, **kwargs):
            input_set_dict = {}
            output_dict = {}
            rng = kwargs.pop('rng', None)
            rngs = jrandom.split(rng, n_layers) if rng is not None else (None,) * n_layers
            
            model_inputs = inputs
            
            output = None
            for l, (fun, param, name, input_name, is_output, rng) in enumerate(zip( self.__apply_funs,
                                                                                    params,
                                                                                    self.__names,
                                                                                    self.__input_names,
                                                                                    self.__is_output,
                                                                                    rngs)):
                if l == 0:
                    layer_inputs = model_inputs
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
                if is_output or (l == len(self.__apply_funs) - 1):
                    assert(name != None)
                    output_dict[name] = output  # for output
            return output_dict
        
        # return 2 functions
        return init_fun, apply_fun


