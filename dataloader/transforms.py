from __future__ import division
import torch
import math
import random
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageEnhance
import collections
import scipy.ndimage.interpolation as itpl
import scipy.misc as misc
import types
import warnings

def _is_tensor_image(image):
    return torch.is_tensor(image) and image.ndimension() == 3

def _is_pil_image(image):
    if accimage is not None:
        return isinstance(image, (Image.Image, accimage.Image))
    else:
        return isinstance(image, Image.Image)

def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for i in self.transforms:
            image = i(image)
        return image


class ToTensor(object):
    def __call__(self, image):
        if not(_is_numpy_image(image)):
            raise TypeError('image must be ndarray. Got {}'.format(type(image)))

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = torch.from_numpy(image.copy())
            elif image.ndim == 3:
                image = torch.from_numpy(image.transpose((2, 0, 1)).copy())
            else:
                raise RuntimeError('image must be ndarray with 3 or 2 dimensions. Got {}'.format(image.ndim))
            return image.float()

class Resize(object):
    def __init__(self, dimension, interpolation='nearest'):
        assert isinstance(dimension, int) or isinstance(dimension, float) or \
               (isinstance(dimension, collections.Iterable) and len(dimension) == 2)
        self.dimension = dimension
        self.interpolation = interpolation

    def __call__(self, image):
        if image.ndim == 2:
            return misc.imresize(image, self.dimension, self.interpolation, 'F')
        elif image.ndim == 3:
            return misc.imresize(image, self.dimension, self.interpolation)
        else:
            RuntimeError('image must be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim))

          
class CenterCrop(object):
    def __init__(self, dimension):
        if isinstance(dimension, numbers.Number):
            self.dimension = (int(dimension), int(dimension))
        else:
            self.dimension = dimension

    @staticmethod
    def get_params(image, output_dimension):
        h = image.shape[0]
        w = image.shape[1]
        sh, sw = output_dimension
        m = int(round((h - sh) / 2.))
        n = int(round((w - sw) / 2.))
        return m, n, sh, sw

    def __call__(self, image):
        m, n, h, w = self.get_params(image, self.dimension)
        if not(_is_numpy_image(image)):
            raise TypeError('image should be ndarray. Got {}'.format(type(image)))
        if image.ndim == 3:
            return image[m:m+h, n:n+w, :]
        elif image.ndim == 2:
            return image[m:m + h, n:n + w]
        else:
            raise RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim))

