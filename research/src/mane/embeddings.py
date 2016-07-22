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
    """
    Initialize a basic embedding neural network model with
    settings in kwargs.

    Parameters
    ----------
      kwargs: Dictionary like arguments. Use for convenient.
      
    Behaviros
    ---------
      Create basic object to store neural network parameters.

    """

    # General hyperparameters for embeddings
    self._emb_dim = getattr(kwargs, 'emb_dim')
    self._learning_rate = getattr(kwargs, 'learning_rate')
    self._batch_size = getattr(kwargs, 'batch_size')
    self._neg_samp = getattr(kwargs, 'neg_samp')
    
  def forward(self, **kwargs):
    """
    Build forward pass
    """
    # Input layers for learning embeddings [word2vec]
    graph_in = keras.layers.Input(shape=(None,), np.int32)
    labels_in = keras.layers.Input(shape=(None,2), np.float32)
    embeddings
    
    

