import mxnet as mx

def get_symbol(data, prefix):
    # declare symbol
    #data = mx.sym.Variable("%s_data" % prefix)
    conv1_1 = mx.symbol.Convolution(name='%s_conv1_1' % prefix, data=data , num_filter=64, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu1_1 = mx.symbol.Activation(data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='%s_conv1_2' % prefix, data=relu1_1 , num_filter=64, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu1_2 = mx.symbol.Activation(data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv2_1 = mx.symbol.Convolution(name='%s_conv2_1' % prefix, data=pool1 , num_filter=128, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu2_1 = mx.symbol.Activation(data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='%s_conv2_2' % prefix, data=relu2_1 , num_filter=128, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu2_2 = mx.symbol.Activation(data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv3_1 = mx.symbol.Convolution(name='%s_conv3_1' % prefix, data=pool2 , num_filter=256, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu3_1 = mx.symbol.Activation(data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='%s_conv3_2' % prefix, data=relu3_1 , num_filter=256, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu3_2 = mx.symbol.Activation(data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='%s_conv3_3' % prefix, data=relu3_2 , num_filter=256, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu3_3 = mx.symbol.Activation(data=conv3_3 , act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv4_1 = mx.symbol.Convolution(name='%s_conv4_1' % prefix, data=pool3 , num_filter=512, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu4_1 = mx.symbol.Activation(data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='%s_conv4_2' % prefix, data=relu4_1 , num_filter=512, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu4_2 = mx.symbol.Activation(data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='%s_conv4_3' % prefix, data=relu4_2 , num_filter=512, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu4_3 = mx.symbol.Activation(data=conv4_3 , act_type='relu')
    pool4 = mx.symbol.Pooling(data=relu4_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv5_1 = mx.symbol.Convolution(name='%s_conv5_1' % prefix, data=pool4 , num_filter=512, dilate=(1,1), pad=(1,1), kernel=(3,3), stride=(1,1), workspace=512)
    relu5_1 = mx.symbol.Activation(data=conv5_1 , act_type='relu')

    return relu2_2
