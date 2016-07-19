"""Motif aware graph embedding model
"""
# Coding: utf-8
# File name: mane.py
# Created: 2016-07-19
# Description: Main file to run the model.
## v0.0: File created. Add argparse

import cPickle as pickle
from numpy import argmax
import argparse

# Parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', 
                    help = 'learning rate for optimizer.',
                    type = float, default = 0.1)
parser.add_argument('--embedding_size', 
                    help = 'output embedding vector dimension.',
                    type = int, default = 100)
parser.add_argument('--window_size',
                    help = 'sampling diameter for walk sequence.',
                    type = int, default = 5)
parser.add_argument('--rd_walk_length',
                    help = 'length of random walk.',
                    type = int, default = 128)
parser.add_argument('--mt_walk_length',
                    help = 'length of motif walk.',
                    type = int, default = 128)
parser.add_argument('--batch_size',
                    help = 'batch size for training neural net.',
                    type = int, defautl = 100)
parser.add_argument('--save_model_to',
                    help = 'file location to save the model',
                    type = str, default = 'mane.model')
args = parser.parse_args()

def acc(tpy, ty):
  """
  Compute accuracy of softmax output tpy and
  true one-hot encoding label ty.

  Parameters
  ----------
    tpy: Vector of probability for label.
    ty: One-hot vector of true label.

  Returns
  -------
    Fraction of correct classification.
  """
  return (argmax(tpy, axis=1) == argmax(ty, axis=1)).sum() 
                                 * 1.0 / tpy.shape[0]

# Load the data
