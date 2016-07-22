"""Neural network embedding model.
"""
# Coding: utf-8
# File name: embeddings.py
# Created: 2016-07-21
# Description

from __future__ import division
from __future__ import print_function

import theano
import keras

from theano import tensor as T

class embeddings_net():
  """
  Contain computation graph for embedding operations.
  The basic default model is similar to deepwalk with negative
  node sampling.
  """
  def __init__(self, **kwargs):
    self._emb_dim = getattr(kwargs, 'emb_dim')
    self._learning_rate = getattr(kwargs, 'learning_rate')


    # Extra setting
    for key, val in kwargs.iteritems():
      setattr(self, key, val)

  def forward(self, **kwargs):
    """
    Build forward pass
    """
    # Input layers for learning embeddings [word2vec]
    graph_in = keras.layers.Input(shape=(None,), np.int32)
    labels_in = keras.layers.Input(shape=(None,2), np.float32)
    embeddings
    
    

