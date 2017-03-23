import sys
sys.path.insert(0, "../mxnet/python")

import mxnet as mx
import numpy as np

import os
import argparse
import logging

import data_processing
import dcnn

parser = argparse.ArgumentParser(description='sketch inversion')

parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--long-edge', type=int, default=96,
                    help='height of the image')

args = parser.parse_args()

# Choose which CPU or GPU card to use
dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
ctx = dev

# Params
long_edge = args.long_edge

# Load data
dir_sketch = "data/sketch/"
file_sketch = os.listdir(dir_sketch)
file_sketch.sort()
training = int(0.7*len(file_sketch)) # 70%
num_sketch = int(len(file_sketch) - training) # 30%
logging.info("Num of sketches: %d" % num_sketch)

# Init sketch
sketch_np = data_processing.PreprocessSketchImage(dir_sketch + file_sketch[training], long_edge)
logging.info("load the sketch image, size = %s", sketch_np.shape[2:])
dshape = sketch_np.shape
clip_norm = 0.05 * np.prod(dshape)

# Load pretrained params
gens = dcnn.get_module("g0", dshape, ctx)
gens.load_params("model/0265-0011975-sketch-vgg16.params")

# Testing 
logging.info('Start testing arguments %s', args)
for idx in range(num_sketch):
    # Load sketch
    sketch_data = []
    path_sketch = dir_sketch + file_sketch[training+idx]
    sketch_np = data_processing.PreprocessSketchImage(path_sketch, long_edge)
    mySketch = mx.nd.array(sketch_np)
    sketch_data.append(mySketch)

    gens.forward(mx.io.DataBatch([sketch_data[-1]], [0]), is_train=False)
    new_img = gens.get_outputs()[0]
    data_processing.SaveImage(new_img.asnumpy(),
                              "output/test/out_%s" % file_sketch[training+idx])
