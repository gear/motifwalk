# Origin: https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# Coding: utf-8
# Filename mnist_tensorboard.py - Python 2.7
# Author: Hoang NT
# Created: 2016-06-28 v0.0
# Description: 
## v0.0: Simple MNIST classifier which displays summaries in TensorBoard

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, use fake data for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flag.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data.')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()
  
  # Create multilayer model
  
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, -1])
    tf.image_summary('input', image_shaped_input, 10)
    
  # We can't initialize these variables to 0 - the network will get stuck
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('stddev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Code for creating simple neural net layer"""
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = weights_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activation', preactivate)
      activations = act(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')
  
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accura
