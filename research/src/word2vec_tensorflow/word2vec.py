"""Multi-threaded word2vec mini-batched skip-gram model.
"""
# Origin: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/embedding/word2vec.py
# Coding: utf-8
# Filename: word2vec.py - Python 2.7
# Created: 2016-07-07 v0.0
# Description: This model is proposed in ICLR 2013. arvix:1301.3781
## v0.0: File copied from origin for learning purpose.

from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange # Faster range iteration

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and " 
                    "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "E.g. https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100, 
                     "Negative samples per training examples processed per step "
                      "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS

class Option(object):
  """Options used by word2vec model. Defined by the flags."""

  def __init__(self):
    # Model options
    
    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options

    # Training text file
    self.train_data = FLAGS.train_data

    # Number of negative samples per example
    self.num_samples = FLAGS.num_neg_samples

    # The inital learning rate
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epoch_to_train = FLAGS.epochs_to_train

    # Concurrent training steps
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size
    
    # The number of words to predict to the left and right of the target word
    self.window_size = FLAGS.window_size

    # The minimum number of word occurences for it to be included in the 
    # vocabulary
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval)
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries
    self.save_path = FLAGS.save_path

    # Evaluate options

    # The text file for evaluation
    self.eval_data = FLAGS.eval_data

class Word2Vec(object):
  """Word2Vec model (Skipgram)."""
  
  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()
    self._read_analogies()

  def _read_analogies(self):
    """Reads through the analogy question file.

    Returns
    -------

    
