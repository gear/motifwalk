"""Neural network embedding model.
"""
# Coding: utf-8
# File name: embeddings.py
# Created: 2016-07-21
# Description:
# v0.0: File created. Simple Sequential model.
# v0.1: Change to functional API model.
# v0.2: Test use fit_generator for basic model.
# v1.0: Change architecture for xentropy and negative sampling.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import keras
import theano
import math

# Import keras modules
from keras.models import Model
from keras.layers import Input, Merge, Reshape, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras import initializations as init

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
    # __init__

    def __init__(self, model=None, graph=None, epoch=1,
                 name='EmbeddingNet', emb_dim=128,
                 neg_samp=5, num_skip=5, num_walk=5,
                 walk_length=80, window_size=10,
                 walk_per_batch=5):
        """
        Initialize a basic embedding neural network model with
        settings in kwargs.

        Parameters
        ----------
          model: Keras Model instance.
          graph: Graph instance contains graph adj list.
          epoch: Number of pass for each batch.
          name: Name of the model.
          emb_dim: Embedding size.
          neg_samp: Number of negative samples for each target.
          num_skip: Number of possitive samples for each target.
          num_walk: Number of walk performed.
          walk_length:
          window_size: Skipgram window for generating +data.
          walk_per_batch: Number of walks per batch.

        Behavior
        --------
          Create basic object to store neural network parameters.
        """
        # General hyperparameters for embeddings
        self._emb_dim = emb_dim
        self._epoch = epoch
        self._neg_samp = neg_samp
        self._num_skip = num_skip
        self._num_walk = num_walk
        self._walk_length = walk_length
        self._window_size = window_size
        self._walk_per_batch = walk_per_batch

        # Status flags
        self._built = False
        self._trained = False

        # Data
        self._graph = graph

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
        input_dim = max(self._graph.keys())+1
        target_in = Input(batch_shape=(None, 1),
                          dtype='int32', name='target')
        class_in = Input(batch_shape=(None, 1),
                         dtype='int32', name='class')
        embeddings = Embedding(input_dim=input_dim,
                               output_dim=self._emb_dim,
                               name='node_embeddings', input_length=1,
                               init=self.init_uniform)(target_in)
        nce_weights = Embedding(input_dim=input_dim,
                                output_dim=self._emb_dim,
                                input_length=1,
                                init=self.init_normal, name="nce_weights_embedding")(class_in)
        nce_bias = Embedding(input_dim=input_dim,
                             output_dim=1, name='nce_bias_emb',
                             input_length=1, init='zero')(class_in)
        embeddings = Reshape((self._emb_dim,), name="reshape_node")(embeddings)
        nce_weights = Reshape((self._emb_dim,), name="reshape_weights")(nce_weights)
        nce_bias = Reshape(target_shape=(1,), name="reshape_bias")(nce_bias)
        dot_prod = Merge(mode=row_dot, output_shape=merge_shape,
                         name='row_wise_dot')([embeddings, nce_weights])
        logits = Merge(mode='sum', output_shape=(1,),
                       name='logits')([dot_prod, nce_bias])
        #sigm = Activation('sigmoid', name='label')(logits)
        self._model = Model(input=[target_in, class_in], output=logits)
        self._model.compile(loss=loss_ns, optimizer=optimizer,
                            name='EmbeddingNet')
        self._built = True

    def get_weights(self):
        return self._model.get_weights()

    def train(self, mode='random_walk', num_nodes_per_batch=500, batch_size=128, verbose=1):
        """
        Load data and train the model.

        Parameters
        ----------

        Returns
        -------
          None. Maybe weights of the embeddings?

        Behavior
        --------
          Load data in batches and train the model.
        """
        self._trained = True
        # Graph data generator with negative sampling
        shuffle = True
        for i in range(self._num_walk):
            print('Graph pass:', i+1 ,'/', self._num_walk)
            for j in range(math.ceil(len(self._graph)/num_nodes_per_batch)):
                print('Mini batch:', j+1, '/', math.ceil(len(self._graph)/num_nodes_per_batch))
                batch_data = self._graph.gen_walk(mode, num_nodes_per_batch,
                                              self._walk_length,
                                              self._neg_samp,
                                              self._num_skip,
                                              shuffle,
                                              self._window_size)
                (targets,classes), labels, wpb = batch_data
                if wpb == num_nodes_per_batch:
                    shuffle = False
                else:
                    print('Reset')
                    shuffle = True
                self._model.fit([targets, classes], [labels], batch_size=batch_size,
                                nb_epoch=self._epoch, verbose=verbose)

    # init_normal
    def init_normal(self, shape, name=None):
        """
        Custom normal initializer for nce
        embedding. Shrink stddev.
        """
        return init.normal(shape=shape, scale=1 / np.sqrt(self._emb_dim), name=name)
    # init_uniform

    def init_uniform(self, shape, name=None):
        """
        Custom uniform initializer for input
        embedding. Values between 1 and -1.
        """
        return init.uniform(shape=shape, scale=1, name=name)
# === END CLASS EmbeddingNet ===

# >>> HELPER FUNCTIONS <<<

# row_dot


def row_dot(inputs):
    """
    Compute row-element-wise dot
    for input 2D matrices
    with sign for label
    """
    return K.batch_dot(inputs[0], inputs[1], axes=1)

def loss_ns(inputs, labels):
    return -K.sum(K.log(K.sigmoid(labels*inputs)))

def merge_shape(inputs):
    return (inputs[0][0], 1)

# === END HELPER FUNCTIONS ===
