# -*- coding: utf-8 -*-

"""
LeNet-5 like convolutional MNIST model example
with two convolutional layers + two fully connected layers.

DropNeuron is used to regularize the last two fully connected layer

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/


================================How to run this script=================================

1.  Run the following command with Dropout, with keep probability of 50%

    $ python lenet-5.py 0 0 0 0.5 0.01

    A Sample of Summary Statistics

    $ sparsity of w_fc1= 55.1476303412 %
    $ sparsity of w_out= 62.8125 %
    $ Total Sparsity=  888684 / 1610752  =  55.171994199 %
    $ Compression Rate =  1.81251378443
    $ Accuracy without prune: 0.9907
    $ Accuracy with prune: 0.9912
    $ Neuron percentage =  3136 / 3136 = 100.0 %
    $ Neuron percentage =  504 / 512 = 98.4375 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  3650 / 3658 = 99.7813012575 %


2. Add L1 regularisation

    $ python lenet-5.py 0.0002 0 0 0.5 0.01

    This should achieve a test error of around 1%
    Better performance can be achieved under different weight initialisation

    A Sample of Summary Statistics

    $ sparsity of w_fc1= 5.42459293288 %
    $ sparsity of w_out= 51.66015625 %
    $ Total Sparsity=  89744 / 1610752  =  5.5715591227 %
    $ Compression Rate =  17.9482973792
    $ Accuracy without prune: 0.9901
    $ Accuracy with prune: 0.9896
    $ Neuron percentage =  1039 / 3136 = 33.131377551 %
    $ Neuron percentage =  320 / 512 = 62.5 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  1369 / 3658 = 37.4248223073 %
    
3.  Run the following command with DropNeuron

    Use DropNeuron only

    $ python lenet-5.py 0.0001 0 0.0005 1 0.01

    A Sample of Summary Statistics

    $ sparsity of w_fc1= 1.4428586376 %
    $ sparsity of w_out= 16.81640625 %
    $ Total Sparsity=  24028 / 1610752  =  1.49172560394 %
    $ Compression Rate =  67.0364574663
    $ Accuracy without prune: 0.9907
    $ Accuracy with prune: 0.9914
    $ Neuron percentage =  907 / 3136 = 28.9221938776 %
    $ Neuron percentage =  110 / 512 = 21.484375 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  1027 / 3658 = 28.0754510662 %

    Use DropNeuron with Dropout

    This should achieve a test error of around 1%
    Better performance can be achieved under different weight initialisation    

Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
"""

from __future__ import absolute_import
from __future__ import division

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

import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate_ini = 0.001
lambda_l1 = float(sys.argv[1])
lambda_l2 = float(sys.argv[2])
lambda_dropneuron = float(sys.argv[3])
keep_prob = float(sys.argv[4])   # keep_prob \in (0, 1]
threshold = float(sys.argv[5])

# Parameters
training_iters = 500000
batch_size = 64
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

n_hidden_1 = 7*7*64
n_hidden_2 = 512

# Store layers weight & bias
W = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.01))
}

W_prune = W.copy

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=0.01)),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=0.01)),
    'bfc1': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.01))
}


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def model(x, W, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, W['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, W['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, W['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, W['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, W['out']), biases['out'])
    return out


# Add regularizers
def l1(x):
    regularizers = (l1_regularizer(.1)(W['wfc1']) + l1_regularizer(.1)(biases['bfc1']))
    regularizers += (l1_regularizer(.1)(W['out']) + l1_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def l2(x):
    regularizers = (l2_regularizer(.1)(W['wfc1']) + l2_regularizer(.1)(biases['bfc1']))
    regularizers += (l2_regularizer(.1)(W['out']) + l2_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def dropneuron(x):
    regularizers = (lo_regularizer(.1)(W['wfc1'])) + tf.reduce_mean(li_regularizer(.1)(W['wfc1']))
    regularizers += (lo_regularizer(.1)(W['out'])) + tf.reduce_mean(li_regularizer(.1)(W['out']))
    regularizers = x * regularizers
    return regularizers


def prune(x):
    # Due to machine precision, typically, there is no absolute zeros solution.
    # Therefore, we set a very small threshold to prune some parameters:
    # However, the test error is obtained after pruning
    y_noprune = sess.run(x)
    y_noprune = np.asarray(y_noprune)
    low_values_indices = abs(y_noprune) < threshold
    y_prune = np.copy(y_noprune)
    y_prune[low_values_indices] = 0
    return y_noprune, y_prune


def neuron_input(w):
    neuron_left = np.count_nonzero(np.linalg.norm(w, axis=1))
    neuron_total = np.shape(w)[0]
    print "Neuron percentage = ", neuron_left, "/", neuron_total, \
        "=", float(neuron_left)/float(neuron_total)*100, "%"
    return neuron_left, neuron_total


def neuron_output(w):
    neuron_left = np.count_nonzero(np.linalg.norm(w, axis=0))
    neuron_total = np.shape(w)[1]
    print "Neuron percentage = ", neuron_left, "/", neuron_total, \
        "=", float(neuron_left)/float(neuron_total)*100, "%"
    return neuron_left, neuron_total


def neuron_layer(w1, w2):
    neuron_in = np.count_nonzero(np.linalg.norm(w1, axis=0))
    neuron_out = np.count_nonzero(np.linalg.norm(w2, axis=1))
    neuron_left = min(neuron_in, neuron_out)
    neuron_total = np.shape(w1)[1]
    print "Neuron percentage = ", neuron_left, "/", neuron_total, \
        "=", float(neuron_left)/float(neuron_total)*100, "%"
    return neuron_left, neuron_total


# Construct model
pred = model(x, W, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
cost = loss
cost += l1(lambda_l1)
cost += l2(lambda_l2)
cost += dropneuron(lambda_dropneuron)

optimizer = tf.train.AdamOptimizer(
        learning_rate_ini, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            val_loss, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(val_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    accuracy_noprune = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    w_fc1_, w_fc1 = prune(W['wfc1'])
    W_prune['wfc1'] = W['wfc1'].assign(w_fc1, use_locking=False)
    print "w_fc1 =", '\n',  w_fc1, "shape = ", np.shape(w_fc1), '\n'
    w_out_, w_out = prune(W['out'])
    W_prune['out'] = W['out'].assign(w_out, use_locking=False)
    print "w_out =", '\n',  w_out, "shape = ", np.shape(w_out), '\n'
    sess.run(W_prune)
    
    sparsity = np.count_nonzero(w_fc1)
    sparsity += np.count_nonzero(w_out)

    print "sparsity of w_fc1=", \
        float(np.count_nonzero(w_fc1))/float(np.size(w_fc1))*100, "%"
    print "sparsity of w_out=", \
        float(np.count_nonzero(w_out))/float(np.size(w_out))*100, "%"

    num_parameter = np.size(w_fc1)
    num_parameter += np.size(w_out)
    total_sparsity = float(sparsity)/float(num_parameter)

    print "Total Sparsity= ", sparsity, "/", num_parameter, \
        " = ", total_sparsity*100, "%"
    print "Compression Rate = ", float(num_parameter)/float(sparsity)

    accuracy_prune = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print "Accuracy without prune:", accuracy_noprune
    print "Accuracy with prune:", accuracy_prune

    neuron_left_ = 0
    neuron_total_ = 0
    neuron_left, neuron_total = neuron_input(w_fc1)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_fc1, w_out)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_output(w_out)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    print "Total Neuron Percentage = ", \
        neuron_left_, "/", neuron_total_, "=", float(neuron_left_)/float(neuron_total_)*100, "%"

    savemat('result/result_lenet-5.mat',
            {'w_fc1_': w_fc1_,
             'w_out_': w_out_,
             'w_fc1': w_fc1,
             'w_out': w_out,
             'learning_rate': learning_rate_ini,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'lambda_dropneuron': lambda_dropneuron,
             'keep_prob': keep_prob,
             'threshold': threshold,
             'accuracy_prune': accuracy_prune,
             'accuracy_noprune': accuracy_noprune})
