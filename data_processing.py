import numpy as np
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle
import logging
import random
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

def PreprocessSketchImage(path, long_edge, dshape=None):
    img = io.imread(path)
    resized_img = transform.resize(img, (long_edge,long_edge))
    sample = np.asarray(resized_img) * 256
    if len(sample.shape) == 3:
        sample = sample[:,:,0]
    sample = sample[:,:,np.newaxis]

    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 128 # grayscale
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessPhotoImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68  # R
    sample[1, :] -= 116.779 # G
    sample[2, :] -= 103.939 # B

##    sample[0, :] -= 129.1863  # R
##    sample[1, :] -= 104.7624 # G
##    sample[2, :] -= 93.5940 # B
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939

##    img[0, :] += 129.1863
##    img[1, :] += 104.7624
##    img[2, :] += 93.5940
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename, remove_noise=0.02):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)
