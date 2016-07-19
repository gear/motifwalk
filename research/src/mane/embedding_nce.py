"""Neural network to learn network embeddings
"""
# Coding: utf-8
# Filename: embedding_nce.py
# Created: 2016-07-16
# Description: Neural network on keras/theano 
#              framework to learn embeddings
## v0.0: File created

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import theano
import keras

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

class GraphEmbedding(keras.models.Sequential):
  """
  Custom class MotifEmbedding
