# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

================================How to run this script=================================
The specification of Alexnet using TensorFlow is complicated,
therefore we are using modular framework to make the modelling simpler but transparent

In this script, please install 'TFLearn' available at https://github.com/tflearn/tflearn

This is a modification of the example in TFLearn, available at:
https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py

You can run the following command without DropNeuron or Dropout

   $ python alexnet.py 0 0 0.0005 1 0.001

Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from regularizers import *

import sys
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from scipy.io import savemat

import tflearn.datasets.oxflower17 as oxflower17
# import oxflower17 as oxflower17
trainX, trainY = oxflower17.load_data(one_hot=True)

n_class = 17

learning_rate_ini = 0.001
lambda_l1 = float(sys.argv[1])
lambda_l2 = float(sys.argv[2])
lambda_dropneuron = float(sys.argv[3])
keep_prob = float(sys.argv[4])   # keep_prob \in (0, 1]
threshold = float(sys.argv[5])


def prune(x):
    # Due to machine precision, there is no absolute zeros in the solution usually;
    # Therefore, we set a very small threshold to prune some parameters:
    # However, the test error is obtained after pruning
    y = trainer.get_weights(x)
    y = np.asarray(y)
    low_values_indices = abs(y) < threshold
    y[low_values_indices] = 0
    return y

# Define a dnn using Tensorflow
with tf.Graph().as_default():
    # Building 'AlexNet'
    # Placeholders for data and labels
    X = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    Y = tf.placeholder(shape=(None, n_class), dtype=tf.float32)
    # network = input_data(shape=[None, 224, 224, 3],  name='input')
    network = X
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, keep_prob)
    # network = fully_connected(network, 4096, activation='tanh')
    # network = dropout(network, 0.5)
    # network = fully_connected(network, n_class, activation='softmax')
    network_fc1 = fully_connected(network, 4096, activation='tanh')
    network = dropout(network_fc1, keep_prob)
    network_fc2 = fully_connected(network, n_class, activation='softmax')
    network = network_fc2
    fc1_weights = network_fc1.W
    fc2_weights = network_fc2.W

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, Y))

    regularizers = tf.reduce_mean(li_regularizer(.1)(fc1_weights))
    regularizers += tf.reduce_mean(lo_regularizer(.1)(fc1_weights))
    regularizers += tf.reduce_mean(li_regularizer(.1)(fc2_weights))
    regularizers += tf.reduce_mean(lo_regularizer(.1)(fc2_weights))
    loss += lambda_dropneuron * regularizers


    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ini)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(network, 1), tf.argmax(Y, 1)), tf.float32),
        name="acc")

    # Define a train op
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    # Tensorboard logs stored in /tmp/tflearn_logs/. Using verbose level 2.
    trainer = tflearn.Trainer(train_ops=trainop,
                            tensorboard_dir='tflearn_logs/',
                            tensorboard_verbose=2)

    # Training
    # trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY},
    #             n_epoch=10, show_metric=True, run_id='Summaries_example')
    # trainer.fit({X: trainX, Y: trainY}, n_epoch=10, show_metric=True, run_id='Summaries_example')

    trainer.fit(feed_dicts={X: trainX, Y: trainY}, n_epoch=1000, val_feed_dicts=0.1,
                shuffle_all=True, show_metric=True, snapshot_step=200,
                snapshot_epoch=False, run_id='alexnet_oxflowers17')

    # Force the weight to be zeros below a threshold
    w_fc1_weights = prune(fc1_weights)
    print("w_fc1_weights =", '\n',  w_fc1_weights, "shape = ", np.shape(w_fc1_weights), '\n')

    w_fc2_weights = prune(fc2_weights)
    print("w_fc2_weights =", '\n',  w_fc2_weights, "shape = ", np.shape(w_fc2_weights), '\n')

    savemat('result_alexnet.mat', {'w_fc1_weights': w_fc1_weights,
                                   'w_fc2_weights': w_fc2_weights,
                                   'learning_rate_': learning_rate_ini,
                                   'lambda_l1': lambda_l1,
                                   'lambda_l2': lambda_l2,
                                   'lambda_dropneuron': lambda_dropneuron,
                                   'dropout': keep_prob})

    # Run the following command to start tensorboard:
    # >> tensorboard /tmp/tflearn_logs/
    # Navigate with your web browser to http://0.0.0.0:6006/
