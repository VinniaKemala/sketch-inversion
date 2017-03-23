'''
    DCNN + VGG, using mx.module libs
    UPDATE: Add pixel loss, lr_scheduler
'''
import sys
sys.path.insert(0, "../../mxnet/python")

import mxnet as mx
import numpy as np

import os
import argparse
import logging
import time

import basic
import data_processing
import dcnn

parser = argparse.ArgumentParser(description='sketch inversion')

parser.add_argument('--feature-weight', type=float, default=1,
                    help='the weight for feature loss')
parser.add_argument('--pixel-weight', type=float, default=1,
                    help='the weight for pixel loss')
parser.add_argument('--tv-weight', type=float, default=1e-2,
                    help='the magtitute on TV loss')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.9,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--gpu', type=int, default=0,
                    help='the gpu will be used. -1 means using cpu')
parser.add_argument('--long-edge', type=int, default=96,
                    help='height of the image')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='the number of train epochs')
parser.add_argument('--save-epochs', type=int, default=2,
                    help='save the output every n epochs')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-file', type=str, default="log_tr_sketch",
                    help='the name of log file')
parser.add_argument('--log-dir', type=str, default='log/',
                    help='directory of the log file')

args = parser.parse_args()

# logging
kv = mx.kvstore.create(args.kv_store)
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
if 'log_file' in args and args.log_file is not None:
    log_file = args.log_file
    log_dir = args.log_dir
    log_file_full_name = os.path.join(log_dir, log_file)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('start with arguments %s', args)
else:
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

# tv-loss
def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})

# Choose which CPU or GPU card to use
dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
ctx = dev

# params
vgg_params = mx.nd.load("pretrained/vgg16-0000.params")
long_edge = args.long_edge

# Load data (num of sketches and photos must be same)
dir_sketch = "data/sketch/"
file_sketch = os.listdir(dir_sketch)
file_sketch.sort()
num_sketch = len(file_sketch)
logging.info("Num of sketches: %d" % num_sketch)

dir_photo = "data/photo/"
file_photo = os.listdir(dir_photo)
file_photo.sort()
num_photo = int(num_sketch)
logging.info("Num of photos: %d" % num_photo)

# Init sketch
sketch_np = data_processing.PreprocessSketchImage(dir_sketch + file_sketch[0], long_edge) # dicek perlu apa gak
logging.info("load the sketch image, size = %s", sketch_np.shape[2:])
dshape = sketch_np.shape
clip_norm = 0.05 * np.prod(dshape)

# Init photo
photo_np = data_processing.PreprocessPhotoImage(dir_photo + file_photo[0], dshape)
logging.info("load the photo image, size = %s", photo_np.shape[2:])

# Photo module
photo_mod = basic.get_photo_module("photo", dshape, ctx, vgg_params)

# Loss module
loss = basic.get_loss_module("loss", dshape, ctx, vgg_params)
grad_array = []
grad_array.append(mx.nd.ones((1,), ctx) * (float(args.feature_weight)))
grad_array.append(mx.nd.ones((1,), ctx) * (float(args.pixel_weight)))

# Model module
gens = dcnn.get_module("g0", dshape, ctx)

# Adam optim params
optim_args = {'learning_rate': args.lr, 'beta1': 0.9, 'wd': 1e-4,
              'clip_gradient': 5.0}

# Every 20 epochs, the learning rate is be reduced by 10%
if 'lr_factor' in args and args.lr_factor < 1:
    print('Found lr_factor %f' % args.lr_factor)
    optim_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
        step = max(int(num_sketch * 20 * args.lr_factor_epoch), 1),
        factor = args.lr_factor)
    
gens.init_optimizer(optimizer='adam', optimizer_params=optim_args)

# Train 
for i in range(args.num_epochs):
    tic = time.time()
    tot_gnorm = float(0)
    for idx in range(num_sketch):
        loss_grad_array = []
        # Load sketch
        sketch_data = []
        path_sketch = dir_sketch + file_sketch[idx]
        sketch_np = data_processing.PreprocessSketchImage(path_sketch, long_edge)
        sketch_nd = mx.nd.array(sketch_np)
        sketch_data.append(sketch_nd)
        # Load photo
        photo_data = []
        path_photo = dir_photo + file_photo[idx]
        photo_np = data_processing.PreprocessPhotoImage(path_photo, dshape)
        photo_nd = mx.nd.array(photo_np)        
        # Get photo representation (using VGG model)
        photo_mod.forward(mx.io.DataBatch([photo_nd], [0]), is_train=False)
        photo_array = photo_mod.get_outputs()[0].copyto(mx.cpu())
        # Set Photo_nd as a target pixel
        loss.set_params({"target_pixel" : photo_nd, "target_feature" : photo_array,}, {}, True, True)
        # gen_forward
        gens.forward(mx.io.DataBatch([sketch_data[-1]], [0]), is_train=True)
        sketch_data.append(gens.get_outputs()[0].copyto(mx.cpu()))
        # loss forward
        loss.forward(mx.io.DataBatch([sketch_data[-1]], [0]), is_train=True)
        loss.backward(grad_array)
        grad = loss.get_input_grads()[0]
        loss_grad_array = grad.copyto(mx.cpu())

        grad = mx.nd.zeros(sketch_nd.shape)

        tv_grad_executor = get_tv_grad_executor(gens.get_outputs()[0],
                ctx, args.tv_weight)
        tv_grad_executor.forward()        
        # new_grad
        grad[:] += loss_grad_array + tv_grad_executor.outputs[0].copyto(mx.cpu())
        gnorm = mx.nd.norm(grad).asscalar()
        if gnorm > clip_norm:
            #logging.info("Epoch[%d] Image[%d]: Data Grad: %.5f" %\
            #             (i, idx, (gnorm / clip_norm)))
            grad[:] *= clip_norm / gnorm
            tot_gnorm += gnorm / clip_norm
        gens.backward([grad])
        gens.update()

        new_img = gens.get_outputs()[0]

        if (idx+1) % num_sketch == 0:
            ave_gnorm = tot_gnorm / num_sketch
            logging.info("Epoch[%d]: Epoch Data Grad: %.5f" % (i+1, ave_gnorm))

        if ((i+1) % args.save_epochs == 0 and (idx+1) % num_sketch == 0):
            data_processing.SaveImage(new_img.asnumpy(), 'output/train/out_'+str(i+1)+'_'+file_sketch[idx])
            gens.save_params("model/%04d-%07d-sketch.params" % ((i+1), (idx+1)))
            logging.info("Save Params : %04d-%07d-sketch.params" % ((i+1), (idx+1)))

    logging.info("Time elapsed for one epoch in seconds: " + str(time.time()-tic))

