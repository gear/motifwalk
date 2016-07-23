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

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

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
    
  def build_model(self, config=None):
    """
    Build the neural network.
    
    Parameters
    ----------
      config: Dictionary of Keras layers to build. 
              If None, a default embedding model is 
              loaded.

    Returns
    -------
      List of layer instances

    Behavior
    --------
      Add layers to the model. Each model should only
      be built once. Print warning when trying to build
      the built model.
    """
    if self._built:
      print('WARNING: Model was built.'
            ' Performing more than one build...')
    if not config:
      self.add(Dense(32, input_dim=784))
      self.add(Activation('relu'))
      self.add(Dense(1000, input_dim=1000))
    else:
      for layer in config:
        self.add(layer)
    self._built = True

  def compile_model(self, loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy']):
    """
    Compile the model by adding loss functions and
    optimizer. 

    Parameters
    ----------
      loss: String identifier for the loss function.
      optimizer: String identifier for the optimizer.
      metric: A list of metrics setting. Only 'accuracy'
              for now.

    Returns
    -------
      None.

    Behavior
    --------
      Compile the model and add appropriate loss and
      optimization operation. Since this is a simple
      sequential model, the input is the first layer
      and loss function is computed at the last layer.
    """
    if self._compiled:
      print('Model is compiled.')
    else:
      self.compile(loss=loss, optimizer=optimizer, 
                   metrics=metrics)
      self._compiled = True

  def fit_data(x, y, nb_epoch):
    















































