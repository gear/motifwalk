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
from keras.models import Model, Sequential
from keras.layers import Input, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras import backend as K

# Import custom layers
from custom_layers import RowDot

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

    # Inputs and output
    self._target = None
    self._class = None
    self._score = None
  ######################################################################## build
  def build(self, loss=None, optimizer='adam'):
    """
    Build and compile neural net with functional API.
    
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
    target_in = Input(batch_shape=(1,), 
                      dtype='int32', name='target_in')
    class_in = Input(batch_shape=(1,), 
                     dtype='int32', name='class_in')
    # Embedding layers connect to target_in and class_in
    emb_in = Embedding(input_dim=len(self._graph),
                       output_dim=self._emb_dim, 
                       input_length=self._batch_size, 
                       name='emb_in')(target_in)
    reshape_in = Reshape(target_shape=(self._emb_dim,))(emb_in)
    emb_out = Embedding(input_dim=len(self._graph),
                        output_dim=self._emb_dim, 
                        input_length=self._batch_size, 
                        name='emb_out')(class_in)
    reshape_out = Reshape(target_shape=(self._emb_dim,))(emb_out)
    # Elemen-wise multiplication for dot product
    dot_prod = Merge(mode=row_wise_dot, output_shape=(100,1), 
                     name='dot_prod')([reshape_in, reshape_out])
    # Initialize model
    if self._model is None:
      self._model = Model(input=[target_in, class_in], output=dot_prod)
    # Compile model
    if not self._compiled:
      self._model.compile(loss=loss, optimizer=optimizer, name='model')
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
        self._model.fit({'target_in':targets, 'class_in':classes}, 
                        {'dot_prod':labels}, batch_size=self._batch_size, 
                        nb_epoch=1)

  def build_sequential(self, loss=None, optimizer='adam'):
    """
    Build and compile neural net using Sequential model.

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

    # Create blocks for adding layers
    self._model = Sequential()
    emb_in = Sequential()
    emb_out = Sequential()
    
    # Target embedding construct
    emb_in.add(Embedding(input_dim=len(self._graph), output_dim=self._emb_dim,
                         input_length=self._batch_size, name='emb_in_layer'))
    # Class embedding construct
    emb_out.add(Embedding(input_dim=len(self._graph), output_dim=self._emb_dim,
                         input_length=self._batch_size, name='emb_out_layer'))
    # Merge emb_in and emb_out
    merged = Merge([emb_in, emb_out], mode=row_wise_dot, output_shape=(100,1)) #TODO: Quick fix output_shape

    # Final model
    self._model.add(merged)
    self._model.compile(optimizer=optimizer, loss=loss)
    

# === END CLASS EmbeddingNet ===

# >>> HELPER FUNCTIONS <<<

####################################################################### nce_loss 
def nce_loss(y_true, y_pred):
    """
    Custom NCE loss function.
    """
    return -K.log(K.sigmoid(y_pred * y_true)).sum()

def row_wise_dot(inputs):
    """
    Custom function for loss
    """
    a = inputs[0]
    b = inputs[1]
    return K.batch_dot(a,b,axes=[2,2])

# === END HELPER FUNCTIONS ===













