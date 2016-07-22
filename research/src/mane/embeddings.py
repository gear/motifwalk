"""Neural network embedding model.
"""
# Coding: utf-8
# File name: embeddings.py
# Created: 2016-07-21
# Description:
## v0.0: File created. Simple Sequential model.

from __future__ import division
from __future__ import print_function

import keras
import theano

from keras.models import Sequential
from keras.layers import Dense, Activation

class EmbeddingNet(Sequential):
  """
  Contain computation graph for embedding operations.
  The basic default model is similar to deepwalk with negative
  node sampling. This model only perform embedding based
  on graph structure (explain the 'Net' name).
  """

  def __init__(self, name='EmbeddingNet', layers=[], **kwargs):
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
    # Initialize with super proxy
    super(EmbeddingNet, self).__init__(layers=layers, name=name)

    # General hyperparameters for embeddings
    self._emb_dim = getattr(kwargs, 'emb_dim')
    self._learning_rate = getattr(kwargs, 'learning_rate')
    self._batch_size = getattr(kwargs, 'batch_size')
    self._neg_samp = getattr(kwargs, 'neg_samp')

    # Data 
    self._graph = getattr(kwargs, 'graph')
    
  def forward(self):
    """
    Build the neural network for the forward pass.
    """
    self.add(Dense(32, input_dim=784))
    self.add(Activation('relu'))
    

