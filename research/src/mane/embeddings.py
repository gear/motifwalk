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

  def __init__(self, graph=None,
               name='EmbeddingNet', layers=[],
               emb_dim=200, learning_rate=0.01,
               batch_size=100, neg_samp=5, 
               save_file='EmbeddingNet.keras'):
    """
    Initialize a basic embedding neural network model with
    settings in kwargs.

    Parameters
    ----------
      graph: Graph instance contains graph adj list.
      name: Name of the model.
      layers: List of layer instances for model initialization.
      emb_dim: Embedding size.
      learning_rate: Learning rate (lr).
      batch_size: Size of each training batch.
      neg_samp: Number of negative samples for each target.
      save_file: File location to save the model.
      
    Behavior
    --------
      Create basic object to store neural network parameters.

    """
    # Initialize with super proxy
    super(self).__init__(layers=layers, name=name)

    # General hyperparameters for embeddings
    self._emb_dim = emb_dim
    self._learning_rate = learning_rate
    self._batch_size = batch_size
    self._neg_samp = neg_samp
    self._save_file = save_file

    # Status flags
    self._built = False
    self._compiled = False

    # Data 
    self._graph = graph
    
  def build(self, config=None):
    """
    Build the neural network.
    
    Parameters
    ----------
      config: List of Keras layers to build. If None, a
              default embedding model is loaded.

    Returns
    -------
      None.

    Behavior
    --------
      Add 
    """
    
    if not config:
      self.add(Dense(32, input_dim=784))
      self.add(Activation('relu'))

  def compile(self):
    """
    Compile the graph builded
    

