# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# This code is modified based on the official implementation from Goolge TensorFlow
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
# I hereby acknowledge the efforts from TensorFlow authors
#
# The main difference lies in the new function "dropneuron()"



"""
LeNet-5 like convolutional MNIST model example.

DropNeuron is used to regularize the last two fully connected layer

================================How to run this script=================================

1.  Run the following command with DropNeuron

    $ python convnet.py 0.0002 0 0.0005 1 0.01

    This should achieve a test error of around 1%
    Better performance can be achieved under different weight initialisation

    A Sample of Summary of Statistics

    $ sparsity of w_fc1= 1.9799680126 %
    $ sparsity of w_fc2= 36.26953125 %
    $ Total Sparsity=  33648 / 1610752  =  2.08896217419 %
    $ Compression Rate =  47.8706609605
    $ Test error without prune: 1.0%
    $ Test error with with prune: 0.9%
    $ Neuron percentage =  861 / 3136 = 27.4553571429 %
    $ Neuron percentage =  254 / 512 = 49.609375 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  1125 / 3658 = 30.7545106616 %

2.  Run the following command with Dropout, with keep probability of 50%

    $ python convnet.py 0.0002 0 0 0.5 0.01

    This should achieve a test error of around 1%
    Better performance can be achieved under different weight initialisation

    $ sparsity of w_fc1= 5.70018534758 %
    $ sparsity of w_fc2= 66.875 %
    $ Total Sparsity=  94948 / 1610752  =  5.89463803242 %
    $ Compression Rate =  16.9645700805
    $ Test error without prune: 1.3%
    $ Test error with with prune: 1.1%
    $ Neuron percentage =  1571 / 3136 = 50.0956632653 %
    $ Neuron percentage =  422 / 512 = 82.421875 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  2003 / 3658 = 54.756697649 %

Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from regularizers import *

import gzip
import os
import sys
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.io import savemat

learning_rate_base = 0.001
lambda_l1 = float(sys.argv[1])
lambda_l2 = float(sys.argv[2])
lambda_dropneuron = float(sys.argv[3])
keep_prob = float(sys.argv[4])   # keep_prob \in (0, 1]
threshold = float(sys.argv[5])

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 10000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""

  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = np.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=np.float32)
  labels = np.zeros(shape=(num_images,), dtype=np.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

  train_size = train_labels.shape[0]
  n_hidden_1 = IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64
  n_hidden_2 = 512
  n_classes = NUM_LABELS

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([32]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [n_hidden_1, n_hidden_2],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[n_hidden_2]))
  fc2_weights = tf.Variable(
      tf.truncated_normal([n_hidden_2, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


  fc1_weights_prune = fc1_weights
  fc2_weights_prune = fc2_weights


  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, keep_prob, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  def l1(x):
    # L1 regularization for the fully connected parameters.
    regularizers_l1 = (l1_regularizer(.1)(fc1_weights) + l1_regularizer(.1)(fc1_biases)) +\
                    (l1_regularizer(.1)(fc2_weights) + l1_regularizer(.1)(fc2_biases))
    regularizers = x * regularizers_l1
    return regularizers

  def l2(x):
    # L2 regularization for the fully connected parameters.
    regularizers_l2 = (l2_regularizer(.1)(fc1_weights) + l2_regularizer(.1)(fc1_biases)) +\
                      (l2_regularizer(.1)(fc2_weights) + l2_regularizer(.1)(fc2_biases))
    regularizers = x * regularizers_l2
    return regularizers

  def dropneuron(x):
      # Group regularization for dropping neurons in the fully connected layer.
      regularizers_fc_1 = (li_regularizer(.1)(fc1_weights)) + \
                          (lo_regularizer(.1)(fc1_weights))
      regularizers_fc_2 = (li_regularizer(.1)(fc2_weights)) + \
                          (lo_regularizer(.1)(fc2_weights))
      regularizers_dropneuron = regularizers_fc_1 + regularizers_fc_2
      regularizers = x * regularizers_dropneuron
      return regularizers


  def prune(x):
      # Due to machine precision, typically, there is no absolute zeros solution.
      # Therefore, we set a very small threshold to prune some parameters:
      # However, the test error is obtained after pruning
      y_noprune = sess.run(x)
      y_noprune = np.asarray(y_noprune)
      low_values_indices = abs(y_noprune) < threshold
      y_prune = y_noprune
      y_prune[low_values_indices] = 0
      return y_noprune, y_prune

  def neuron_input(w):
    neuron_left = np.count_nonzero(np.linalg.norm(w, axis=1))
    neuron_total = np.shape(w)[0]
    print("Neuron percentage = ", neuron_left, "/", neuron_total,
          "=", float(neuron_left)/float(neuron_total)*100, "%")
    return neuron_left, neuron_total


  def neuron_output(w):
      neuron_left = np.count_nonzero(np.linalg.norm(w, axis=0))
      neuron_total = np.shape(w)[1]
      print("Neuron percentage = ", neuron_left, "/", neuron_total,
            "=", float(neuron_left)/float(neuron_total)*100, "%")
      return neuron_left, neuron_total


  def neuron_layer(w1, w2):
      neuron_in = np.count_nonzero(np.linalg.norm(w1, axis=0))
      neuron_out = np.count_nonzero(np.linalg.norm(w2, axis=1))
      neuron_left = max(neuron_in, neuron_out)
      neuron_total = np.shape(w1)[1]
      print("Neuron percentage = ", neuron_left, "/", neuron_total,
            "=", float(neuron_left)/float(neuron_total)*100, "%")
      return neuron_left, neuron_total


  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  loss += l1(lambda_l1)
  loss += l2(lambda_l2)
  loss += dropneuron(lambda_dropneuron)

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)

  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      learning_rate_base,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

  # Choose different optimizer.
  # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate.minimize(loss, global_step=batch)
  # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False).minimize(loss, global_step=batch)
  # optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, use_locking=False).minimize(loss, global_step=batch)
  optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(loss, global_step=batch)


  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):

      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_error_noprune = error_rate(eval_in_batches(test_data, sess), test_labels)
    # print('Test error without prune: %.1f%%' % test_error_noprune)
    if FLAGS.self_test:
      print('test_error_noprune', test_error_noprune)
      assert test_error_noprune == 0.0, 'expected 0.0 test_error_noprune, got %.2f' % (
          test_error_noprune,)

    w_fc1_, w_fc1 = prune(fc1_weights)
    fc1_weights_prune = fc1_weights.assign(w_fc1, use_locking=False)
    print("w_fc1 =", '\n',  w_fc1, "shape = ", np.shape(w_fc1), '\n')
    w_fc2_, w_fc2 = prune(fc2_weights)
    fc2_weights_prune = fc2_weights.assign(w_fc2, use_locking=False)
    print("w_fc2 =", '\n',  w_fc2, "shape = ", np.shape(w_fc2), '\n')
    sess.run(fc1_weights_prune)
    sess.run(fc2_weights_prune)
    
    sparsity = np.count_nonzero(w_fc1)
    sparsity += np.count_nonzero(w_fc2)

    print("sparsity of w_fc1=",
          float(np.count_nonzero(w_fc1))/float(np.size(w_fc1))*100, "%")
    print("sparsity of w_fc2=",
          float(np.count_nonzero(w_fc2))/float(np.size(w_fc2))*100, "%")

    num_parameter = np.size(w_fc1)
    num_parameter += np.size(w_fc2)
    total_sparsity = float(sparsity)/float(num_parameter)

    print("Total Sparsity= ", sparsity, "/", num_parameter,
          " = ", total_sparsity*100, "%")
    print("Compression Rate = ", float(num_parameter)/float(sparsity))

    test_error_prune = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error without prune: %.1f%%' % test_error_noprune)
    print('Test error with prune: %.1f%%' % test_error_prune)
    if FLAGS.self_test:
      print('test_error_prune', test_error_prune)
      assert test_error_prune == 0.0, 'expected 0.0 test_error_prune, got %.2f' % (
          test_error_prune,)

    neuron_left_ = 0
    neuron_total_ = 0
    neuron_left, neuron_total = neuron_input(w_fc1)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_fc1, w_fc2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_output(w_fc2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    print("Total Neuron Percentage = ",
          neuron_left_, "/", neuron_total_, "=",
          float(neuron_left_)/float(neuron_total_)*100, "%")


    savemat('result/result_convnet_dropout.mat',
            {'w_fc1_': w_fc1_,
             'w_fc2_': w_fc2_,
             'w_fc1': w_fc1,
             'w_fc2': w_fc2,
             'learning_rate': learning_rate,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'lambda_dropneuron': lambda_dropneuron,
             'dropout': keep_prob,
             'threshold': threshold,
             'test_error_noprune': test_error_noprune,
             'test_error_prune': test_error_prune})

if __name__ == '__main__':
  tf.app.run()


