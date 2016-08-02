"""Neural network embedding model.
"""
# Coding: utf-8
# File name: embeddings.py
# Created: 2016-07-21
# Description:
## v0.0: File created. Simple Sequential model.
## v0.1: Change to functional API model.
## v0.2: Test use fit_generator for basic model.
## v1.0: Change architecture for xentropy and negative sampling.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import keras
import theano

# Import keras modules
from keras.models import Model 
from keras.layers import Input, Merge, Reshape, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras import initializations as init

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
               learning_rate=0.01, batch_size=1, 
               neg_samp=5, num_skip=5, num_walk=5,
               walk_length=5, window_size=5,
               samples_per_epoch=100000,
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
    self._trained = False

    # Data 
    self._graph = graph

    # Epoch size
    self._samples_per_epoch = samples_per_epoch

  ######################################################################## build
  def build(self, loss='binary_crossentropy', optimizer='adam'):
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
      The architecture for v1.0 using binary xentropy: (batch size: 100, dim: 200)
      _____________________________________________________________________
      Layer (type)              Output Shape    Param # Connected to       
      =====================================================================
      class (InputLayer)        (100, 1)        0                          
      _____________________________________________________________________
      target (InputLayer)       (100, 1)        0                          
      _____________________________________________________________________
      nce_emb (Embedding)       (100, 1, 200)   807800  class[0][0]        
      _____________________________________________________________________
      target_emb (Embedding)    (100, 1, 200)   807800  target[0][0]       
      _____________________________________________________________________
      nce_bias (Embedding)      (100, 1, 1)     4039    target[0][0]       
      _____________________________________________________________________
      reshape_8 (Reshape)       (100, 200)      0       target_emb[0][0]   
      _____________________________________________________________________
      reshape_9 (Reshape)       (100, 200)      0       nce_emb[0][0]      
      _____________________________________________________________________
      reshape_10 (Reshape)      (100, 1)        0       nce_bias[0][0]     
      _____________________________________________________________________
      row_wise_dot (Merge)      (100, 1)        0       reshape_8[0][0]    
                                                        reshape_9[0][0]    
      _____________________________________________________________________
      logits (Merge)            (100, 1)        0       row_wise_dot[0][0] 
                                                        reshape_10[0][0]   
      _____________________________________________________________________
      activation_1 (Activation) (100, 1)        0       logits[0][0]       
      =====================================================================
      Total params: 1619639
    """
    if self._built:
      print('WARNING: Model was built.'
            ' Performing more than one build...')

    # Input tensors: batch_shape includes batch_size 
    target_in = Input(batch_shape=(self._batch_size,1), 
                      dtype='int32', name='target')
    class_in = Input(batch_shape=(self._batch_size,1), 
                     dtype='int32', name='class')
    # Embedding layers connect to target_in and class_in
    embeddings = Embedding(input_dim=len(self._graph),
                           output_dim=self._emb_dim, 
                           name='embeddings', input_length=1 
                           init=self.init_uniform) (target_in)
    embeddings = Reshape((self._emb_dim,))(embeddings)
    nce_weights = Embedding(input_dim=len(self._graph),
                            output_dim=self._emb_dim, 
                            name='nce_weights', input_length=1
                            init=self.init_normal) (class_in)
    nce_weights = Reshape((self._emb_dim,)) (nce_weights)
    nce_bias = Embedding(input_dim=len(self._graph),
                         output_dim=1, name='nce_bias',
                         input_length=1, init='zero') (class_in)
    # Elemen-wise multiplication for dot product
    dot_prod = Merge(mode=row_wise_dot, output_shape=(1,), 
                     name='row_wise_dot')([embeddings, nce_weights])
    logits = Merge(mode='sum', output_shape=(1,),  
                   name='logits') ([dot_prod, nce_bias])
    # Final output layer. name='label' for data input reason
    sigm = Activation('sigmoid', name='label')  (logits)
    # Initialize model
    self._model = Model(input=[target_in, class_in], output=sigm)
    # Compile model
    self._model.compile(loss=loss, optimizer=optimizer, name='EmbeddingNet')
    self._built = True
    
  ######################################################################## train
  def train(self, mode='random_walk', num_true=1, 
            shuffle=True, verbose=2, distort=0.75,
            threads=1):
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
    self._trained = True
    # Graph data generator with negative sampling
    data_generator = self._graph.gen_walk(mode,
                                 self._walk_length,
                                 self._num_walk,
                                 num_true,
                                 self._neg_samp,
                                 self._num_skip,
                                 shuffle,
                                 self._window_size,
                                 distort)
    self._model.fit_generator(data_generator,
                              samples_per_epoch=self._samples_per_epoch,
                              nb_epoch=self._epoch) # TODO: nb_worker on gtx

  ################################################################## init_normal
  def init_normal(self, shape, name=None):
    """
    Custom normal initializer for nce
    embedding. Shrink stddev.
    """
    return init.normal(shape=shape, scale=1/np.sqrt(self._emb_dim))
  ################################################################# init_uniform
  def init_uniform(self, shape, name=None):
    """
    Custom uniform initializer for input
    embedding. Values between 1 and -1.
    """
    return init.uniform(shape=shape, scale=1, name=name)
# === END CLASS EmbeddingNet ===

# >>> HELPER FUNCTIONS <<<

######################################################################## row_dot
def row_dot(inputs):
    """
    Compute row-element-wise dot
    for input 2D matrices
    """
    a = inputs[0]
    b = inputs[1]
    return K.batch_dot(a,b,axes=[1,1])

# === END HELPER FUNCTIONS ===
