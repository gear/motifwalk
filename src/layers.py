"""coding=utf-8
python=3.5.2
author=gear
func='keras like layers for dnn, copied from gcn.
      github.com/tkipf/gcn'
"""

import tensorflow as tf
import keras as k
from collections import defaultdict as dd


flags = tf.app.flags
FLAGS = flags.FLAGS

# global layer ID dictionary keeping number of layer types
_LAYER_TYPES = dd(int)


def get_layer_count(layer_type=''):
    """Return the number of `layer_type` in the network plus one.
    This function help to assign unique names to layers in a cnn.

    Example:
        # _LAYER_TYPES = {'dense': 1}
        get_layer_count('dense')  # Return 2
    """
    _LAYER_TYPES[layer_type] += 1
    return _LAYER_TYPES[layer_type]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)

def dot(x, y):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if isinstance(x, tf.SparseTensor) and isinstance(y, tf.SparseTensor):
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x,y)
    return res

class Layer(object):
    """Base layer class."""