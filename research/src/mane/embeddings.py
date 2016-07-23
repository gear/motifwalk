"""Neural network embedding model.
"""
# Coding: utf-8
# File name: embeddings.py
# Created: 2016-07-21
# Description:
## v0.0: File created. Simple Sequential model.
## v0.1: Change to functional API model.

from __future__ import division
from __future__ import print_function

import numpy as np
import keras
import theano

# Import keras modules
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras import backend as K

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

class EmbeddingNet():
  """
  Contain computation graph for embedding operations.
  The basic default model is similar to deepwalk with negative
  node sampling. This model only perform embedding based
  on graph structure (explain the 'Net' name).
  """

  def __init__(model=None, graph=None,
               name='EmbeddingNet', emb_dim=200, 
               learning_rate=0.01, batch_size=100, 
               neg_samp=5, save_file='EmbeddingNet.keras'):
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
    # Initialize with default super proxy
    super(self).__init__()

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

    # Neural net
    self._model = model

  def nce_loss(y_true, y_pred):
    """
    Custom NCE loss function.
    """
    return -K.log(K.sigmoid(y_pred.sum() * y_true).sum())
    
  def build(self, loss=self.nce_loss, optimizer='adam'):
    """
    Build and compile neural net.
    
    Parameters
    ----------
      loss: Loss function (Keras objectives). 
      optimizer: String identifier for Keras optimizer.

    Returns
    -------
      None.

    Behavior
    --------
      Construct neural net. Set built flag to True.
    """
    if self._built:
      print('WARNING: Model was built.'
            ' Performing more than one build...')
    assert loss is None, 'Must provide a loss function.'
    assert optimizer is None, 'Must provide optimizer.'

    # Input tensors
    target_in = Input(shape=(self._batch_size, None), dtype='int32')
    class_in = Input(shape=(self._batch_size, None), dtype='int32')
    label_in = Input(shape=(self._batch_size, 1), dtype='floatX')

    # Embedding layers connect to target_in and class_in
    emb_in = Embedding(output_dim=self._emb_dim, input_dim=len(self._graph),
                       input_length=self._batch_size)(target_in)
    emb_out = Embedding(output_dim=self._emb_dim, input_dim=len(self._graph),
                        input_length=self._batch_size)(target_out)

    # Elemen-wise multiplication for dot product
    merge_emb = merge([emb_in, emb_out], mode='mul')
    dot_prod = merge_emb.sum(axis=1)
  
    # Initialize model
    if self._model is not None:
      self._model = Model(input=[target_in, class_in, label_in], output=dot_prod)

    # Compile model
    self._model.compile(loss=loss, optimizer=optimizer)
    
    

  def train(self, loss='categorical_crossentropy', 
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

    










































