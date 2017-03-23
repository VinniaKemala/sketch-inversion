'''
    DCNN + res block + BN
'''

import sys
sys.path.insert(0, "../mxnet/python")

import mxnet as mx
import numpy as np

def Conv(data, num_filter, kernel, pad, stride=(1,1), conv_type=0, out=False):
    if conv_type == 0: # With activation
        sym = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False)
        sym = mx.sym.BatchNorm(sym, fix_gamma=True)
    	if out == False:
            sym = mx.sym.Activation(sym, act_type='relu')
            #sym = mx.sym.LeakyReLU(sym, act_type="leaky")	    
    	else:
            sym = mx.sym.Activation(sym, act_type="tanh")
        return sym
    elif conv_type == 1: # With no activation
        sym = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False)
        sym = mx.sym.BatchNorm(sym, fix_gamma=True)
        return sym

def Deconv(data, num_filter, kernel, pad, stride=(2, 2)):
    sym = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    sym = mx.sym.BatchNorm(sym, fix_gamma=True)
    sym = mx.sym.Activation(sym, act_type='relu')
    #sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    return sym

def res_block(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data
        conv_res1 = Conv(data, num_filter, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=0)        
        conv_res2 = Conv(conv_res1, num_filter, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=1)
        new_data = identity_data + conv_res2
        sym = mx.sym.Activation(new_data, act_type='relu')
        #sym = mx.sym.LeakyReLU(new_data, act_type="leaky")
        return sym
    else:        
        conv_res1 = Conv(data, num_filter, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=0)
        conv_res2 = Conv(conv_res1, num_filter, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=1)
        # adopt project method in the paper when dimension increased
        project_data = Conv(data, num_filter, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=1)
        new_data = project_data + conv_res2
        sym = mx.symbol.Activation(new_data, act_type='relu')
        #sym = mx.sym.LeakyReLU(new_data, act_type="leaky")
        return sym

def res_net(data, n, dim_match=False):
    for i in range(n):
        data = res_block(data, 128, dim_match)
    return data
    
def get_symbol(prefix, im_hw):
    data = mx.sym.Variable("%s_data" % prefix)
    # 3 conv
    conv1 = Conv(data, 32, kernel=(9,9), pad=(4,4), conv_type=0)
    conv2 = Conv(conv1, 64, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=0)
    conv3 = Conv(conv2, 128, kernel=(3,3), pad=(1,1), stride=(1,1), conv_type=0)
    # 5 Residual blocks
    resnet1 = res_net(conv3, n=1, dim_match=True)
    resnet2 = res_net(resnet1, n=4, dim_match=False)
    # 2 deconv
    deconv1 = Deconv(resnet2, 64, kernel=(3,3), pad=(1,1), stride=(1,1))
    deconv2 = Deconv(deconv1, 32, kernel=(3,3), pad=(1,1), stride=(1,1))
    # last conv
    out = Conv(deconv2, 3, kernel=(9,9), pad=(4,4), conv_type=0, out=True)
    raw_out = (out * 128) + 128
    norm = mx.sym.SliceChannel(raw_out, num_outputs=3)
    r_ch = norm[0] - 128
    g_ch = norm[1] - 128
    b_ch = norm[2] - 128
    norm_out = 0.4 * mx.sym.Concat(*[r_ch, g_ch, b_ch]) + 0.6 * data
    return norm_out

def get_module(prefix, dshape, ctx, is_train=True): 
    sym = get_symbol(prefix, dshape[-2:])
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    if is_train:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=True, inputs_need_grad=True)
    else:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False, inputs_need_grad=False)
    #mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_params(initializer=mx.init.Xavier(
        rnd_type="gaussian", factor_type="in", magnitude=2.))
    return mod  
