"""Simple word2vec implementation on Tensorflow
"""
# Origin: https://github.com/tensorflow/blob/r0.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# Coding: utf-8
# Filename: word2vec_basic.py - Python 2.7
# Created: 2016-06-27 v0.0
# Description:
## v0.0: Basic skip-gram model with Tensorflow

from __future__ import division
from __future__ import print_function

# high performance containers: namedtuples, defaultdict, OrderedDict
import collections 
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib, xrange
import tensorflow as tf

__author__ = "Hoang NT"

# Step 1: Download data and save to datapath folder
url = 'http://mattmahoney.net/dc/'
datapath = './data/'
def maybe_download(filename, expected_bytes):
  """
  Check and download data file.
  
  Parameters
  ----------
  filename: Name of the data file
    If file is not found in the datapath folder, it
    will be downloaded there.
  
  expected_bytes: Expected data file size
    After downloading, the file will be checked
    if it matches the expected size.

  Example
  -------
  Download the words corpus from provided url
  
  >>> maybe_download('./data/text8.zip', 31344016)
  Found and verified ./data/text8.zip 

  """
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
  """
  Extract the file and read as list of words.

  Parameter
  ---------

  filename: Name/location of the zipfile
    filename is the location of the zip file that
    will be extraced and read.

  """
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
  """
  Rebuild dataset as: 2 dictionaries mapping from index to
  word and vice versa, data that contain the original text
  but in index form, and a count that maps from word to its
  count in the corpus.

  Parameter
  ---------

  words: Input text
    List of words from the input data.
  """
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
  """
  Generate data for training.

  Parameters
  ----------

  batch_size: Number of samples
    Number of samples in the generated batch. Each sample
    is a pair of words that is close to each other.

  num_skips: Number of skips in skipgram model
    Number of word pairs generated per target word.

  skip_window: Skipgram model window
    This window defines the range (to the left and right) at
    which words are selected randomly for the target word in
    the center of the skip_window.

  Example
  -------

  >>> generate_batch(20, 4, 5)
  Return 2 arrays each has 20 elements of word index:
    batch: 20 word indices of target words
    labels: 20 word indices of context words
  """
  # global keyword gives this function access to global variable data_index
  global data_index
  assert batch_size % num_skips == 0 
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  # Create a double-ended queue (both stack and queue) for word buffer
  # maxlen - keeping a fixed sliding window  
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    # Shift the skipgram window to the left by 1
    buffer.append(data[data_index])
    # Increase data_index for next shift
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    # target label at the center of the buffer   
    target = skip_window 
    # avoid the target word and later selected words
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      # batch is the same word for current num_skip
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

# Create batch of 8 and print its data
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i,0], reverse_dictionary[labels[i,0]])

# Step 4: Build and train a skip-gram model

# Size of training dataset 
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

# Random validation size
valid_size = 16
valid_window = 100
# Choose valid_size elements from array arange(valid_window)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

# Create a tensorflow graph instance
graph = tf.Graph()

# Add nodes to the created graph
with graph.as_default():
  # Placeholder for inputs and labels
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # Constant valid set
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  with tf.device('/cpu:0'):
    # Input matrix init with uniform random vals minval = -1.0, maxval = 1.0
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # Look up [train_inputs] in a list of [embeddings tensors].
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # 
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                                       num_sampled, vocabulary_size))

  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings =  tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  init = tf.initialize_all_variables()

num_steps = 100001

with tf.Session(graph=graph) as session:
  init.run()
  print("Initialized")
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
    final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18,18))
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x,y)
    plt.annotate(labels,
                 xy=(x,y),
                 xytext=(5,2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Install sklearn and matplotlib.")

# For importing functions to test
def main():
  print('This is main!')
if __name__ == "__main__":
  main()
