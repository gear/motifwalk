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
from keras.layers import Input, merge, Lambda
from keras.layers.embeddings import Embedding
from keras import backend as K

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS EmbeddingNet <<<
class EmbeddingNet():
  """
  Contain computation graph for embedding operations.
  The basic default model is similar to deepwalk with negative
  node sampling. This model only perform embedding based
  on graph structure (explain the 'Net' name).
  """
  ##################################################################### __init__
  def __init__(self, model=None, graph=None, epoch=10,
               name='EmbeddingNet', emb_dim=200, 
               learning_rate=0.01, batch_size=100, 
               neg_samp=5, num_skip=5, num_walk=5,
               walk_length=5, window_size=5,
               save_file='EmbeddingNet.keras'):
    """
    Initialize a basic embedding neural network model with
    settings in kwargs.

    Parameters
    ----------
      model: Keras Model instance.
      graph: Graph instance contains graph adj list.
      epoch: Number of pass through the whole network.
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
    # General hyperparameters for embeddings
    self._emb_dim = emb_dim
    self._learning_rate = learning_rate
    self._epoch = epoch
    self._batch_size = batch_size
    self._neg_samp = neg_samp
    self._save_file = save_file
    self._num_skip = num_skip
    self._num_walk = num_walk
    self._walk_length = walk_length
    self._window_size = window_size

    # Status flags
    self._built = False
    self._compiled = False

    # Data 
    self._graph = graph

    # Neural net
    self._input_tensors = []
    self._output_tensors = []
    self._model = model
  ######################################################################## build
  def build(self, loss=None, optimizer='adam'):
    """
    Build and compile neural net.
    
    Parameters
    ----------
      loss: Loss function (String or Keras objectives).
      optimizer: Keras optimizer (String or object).

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
    if loss is None:
      loss = nce_loss

    # Input tensors: shape doesn't include batch_size
    target_in = Input(shape=(1,), dtype='int32', name='target_in')
    class_in = Input(shape=(1,), dtype='int32', name='class_in')
    # Embedding layers connect to target_in and class_in
    emb_in = Embedding(output_dim=self._emb_dim, input_dim=len(self._graph),
                       input_length=self._batch_size, name='emb_in')(target_in)
    emb_out = Embedding(output_dim=self._emb_dim, input_dim=len(self._graph),
                       input_length=self._batch_size, name='emb_out')(class_in)
    # Elemen-wise multiplication for dot product
    elem_mul = merge([emb_in, emb_out], mode='mul')
    dot_prod = Lambda(reduce_sum, output_shape=(100,1))(elem_mul)
    # Initialize model
    if self._model is None:
      self._model = Model(input=[target_in, class_in], output=dot_prod)
    # Compile model
    if not self._compiled:
      self._model.compile(loss=loss, optimizer=optimizer)
  ######################################################################## train
  def train(self, mode='random_walk', num_true=1, shuffle=True, distort=0.75):
    """
    Load data and train the model.

    Parameters
    ----------
      mode: Data generation mode: 'random_walk' or 'motif_walk'.

    Returns
    -------
      None. Maybe weights of the embeddings?

    Behavior
    --------
      Load data in batches and train the model.
    """
    # Graph data generator with negative sampling
    data_generator = self._graph.sample_walk_with_negative(mode,
                                 self._walk_length,
                                 self._num_walk,
                                 num_true,
                                 self._neg_samp,
                                 self._num_skip,
                                 shuffle,
                                 self._window_size,
                                 self._batch_size,
                                 distort)
    # TODO: Now using makeshift reshape - Fix data generation later
    for _ in xrange(self._epoch):
      for targets, classes, labels in data_generator:
        targets = targets[np.newaxis].T
        classes = classes[np.newaxis].T
        labels = labels[np.newaxis].T
        print(targets.shape)
        print(classes.shape)
        print(labels.shape)
        self._model.fit([targets, classes], labels, 
                        batch_size=self._batch_size, nb_epoch=1)

# === END CLASS EmbeddingNet ===

# >>> HELPER FUNCTIONS <<<

####################################################################### nce_loss 
def nce_loss(y_true, y_pred):
    """
    Custom NCE loss function.
    """
    return -K.log(K.sigmoid(y_pred * y_true)).sum()

# === END HELPER FUNCTIONS ===

def reduce_sum(x):
  return K.sum(x, axis=2)


























