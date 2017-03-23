import sys
sys.path.insert(0, "../../mxnet/python/")

import mxnet as mx
import numpy as np
import model_vgg as vgg

class PretrainedInit(mx.init.Initializer):
    def __init__(self, prefix, params, verbose=False):
        self.prefix_len = len(prefix) + 1
        self.verbose = verbose
        self.arg_params = {k : v for k, v in params.items() if k.startswith("arg:")}
        self.aux_params = {k : v for k, v in params.items() if k.startswith("aux:")}
        self.arg_names = set([k[4:] for k in self.arg_params.keys()])
        self.aux_names = set([k[4:] for k in self.aux_params.keys()])

    def __call__(self, name, arr):
        key = name[self.prefix_len:]
        del_name = ["flatten_0","fc6_weight","fc6_bias","relu6","drop6","fc7_weight","fc7_bias",
                    "relu7","drop7","fc8_weight","fc8_bias"]
        if key in self.arg_names:
            if self.verbose:
                print("Init %s" % name)
            for delete in del_name:
                if key == delete:
                    print("Delete arg: %s" % key)
                    del self.arg_params["arg:" + key]
            self.arg_params["arg:" + key].copyto(arr)            
        elif key in self.aux_params:
            if self.verbose:
                print("Init %s" % name)
            for delete in del_name:
                if key == delete:
                    print("Delete aux: %s" % key)
                    del self.aux_params["aux:" + key]
            self.aux_params["aux:" + key].copyto(arr)
        else:
            print("Unknown params: %s, init with 0" % name)
            arr[:] = 0.


def get_loss(pred_img, pred_feature, num_channels):
    target_pixel = mx.sym.Variable("target_pixel")
    pixel_loss = mx.sym.sum(mx.sym.square(target_pixel - pred_img))
    
    cvar = mx.sym.Variable("target_feature")
    feature_loss = (1./num_channels) * mx.sym.sum(mx.sym.square(cvar - pred_feature))
    return pixel_loss, feature_loss

def get_photo_module(prefix, dshape, ctx, params):
    data = mx.sym.Variable("%s_data" % prefix)
    sym = vgg.get_symbol(data, prefix) # c
    init = PretrainedInit(prefix, params)
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False)
    mod.init_params(init)
    return mod

def get_loss_module(prefix, dshape, ctx, params):
    input_shape = {"%s_data" % prefix : dshape}
    pred_img = mx.sym.Variable("%s_data" % prefix)
    pred_feature = vgg.get_symbol(pred_img, prefix) 
    _, output_shape, _= pred_feature.infer_shape(loss_data=dshape)
    shape = output_shape[0]
    num_channels = shape[1]
    pixel_loss, feature_loss = get_loss(pred_img, pred_feature, num_channels)
    sym = mx.sym.Group([pixel_loss, feature_loss])
    init = PretrainedInit(prefix, params)
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    mod.bind(data_shapes=[("%s_data" % prefix, dshape)],
             for_training=True, inputs_need_grad=True)
    mod.init_params(init)
    return mod
