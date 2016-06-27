# Origin: https://github.com/tensorflow/blob/r0.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# Coding: utf-8
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
  # Using with-as to forget about cleaning up opened files after use
  with zipfile.ZipFile(filename) as f:
    # tf.compat.as_str(.) : converting input to string - use for compatibility between PY2 and PY3.
    # f.namelist(): return list of archive members by name.
    # f.read(.): return bytes of the filename input from the archive.
    # split(): split string to list on whitespace (default).
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
words = read_data(filepath)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rate words with UNK token.
vocabulary_size = 50000
def build_dataset(words):
  # count store list of tuples: (word, count)
  count = [['UNK', -1]]
  # extend(.): append a given list to self
  # most_common(.): return list of n most common elements as tuple.
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  # create a dictionary mapping word to its count ranking dictionary['the'] = 1 - most common
  for word, _ in count:
    dictionary[word] = len(dictionary)
  # count number of discarded words and store the dictionary index for each word index in data
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  # reverse dictionary mapping from word to its count ranking 
  # zip(.): create list of tuples from each provided list. E.g.
  #   [( list1[0], list2[0] ), ( list1[1], list2[1] ) ...]
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words # reduce memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  # global keyword gives this function access to global variable data_index
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  # Create a double-ended queue (both stack and queue) for word buffer
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

if __name__ == "__main__":
  main()
