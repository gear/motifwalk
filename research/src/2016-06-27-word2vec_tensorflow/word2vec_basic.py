# Origin: https://github.com/tensorflow/blob/r0.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# Filename: word2vec_basic.py - Python 2.7
# Author: Hoang NT
# Created: 2016-06-27 v0.0

from __future__ import division
from __future__ import print_function

import collections # high performance containers: namedtuples, defaultdict, OrderedDict
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib, xrange
import tensorflow as tf

# Step 1: Download data and save to datapath folder
url = 'http://mattmahoney.net/dc/'
datapath = './data/'
def maybe_download(filename, expected_bytes):
  """Check and download a file"""
  filepath = datapath + filename
  if not os.path.exists(filepath):
    # urlretrieve returns a tuple of saved filepath and info() of the downloaded file
    filepath, _ = urllib.request.urlretrieve(url+filename, filepath)
  statinfo = os.stat(filepath)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filepath)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filepath + '. Can you get to it with a browser?')
  return filepath
filepath = maybe_download('text8.zip', 31344016)

# Read data into a list of strings
def read_data(filename):
  """Extract the file and read as list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
words = read_data(filepath)
print('Data size', len(words))
