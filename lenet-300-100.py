# -*- coding: utf-8 -*-

""" LeNet-300-100
Using an LeNet-300-100 like network on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

================================How to run this script=================================

1. you can run the following command using DropNeuron

    $ python lenet-300-100.py 0.0001 0  0.0005  0.5 0.01

   A Sample of Summary of Statistics:

    $ sparsity of w_fc1= 9.56335034014 %
    $ sparsity of w_fc2= 11.16 %
    $ sparsity of w_out= 54.5 %
    $ Total Sparsity=  26386 / 266200  =  9.91209616829 %
    $ Compression Rate =  10.0886833927
    $ Accuracy without prune: 0.9813
    $ Accuracy with prune: 0.9817
    $ Neuron percentage =  542 / 784 = 69.1326530612 %
    $ Neuron percentage =  83 / 300 = 27.6666666667 %
    $ Neuron percentage =  61 / 100 = 61.0 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  696 / 1194 = 58.2914572864 %


2. you can run the following command without Regularisation or DropOut

    $ python lenet-300-100.py 0 0 0 0.5 0.01

   A Sample of Summary of Statistics:

    $ sparsity of w_fc1= 83.8405612245 %
    $ sparsity of w_fc2= 84.32 %
    $ sparsity of w_out= 86.1 %
    $ Total Sparsity=  223350 / 266200  =  83.9030803907 %
    $ Compression Rate =  1.19185135438
    $ Accuracy without prune: 0.981
    $ Accuracy with prune: 0.9812
    $ Neuron percentage =  784 / 784 = 100.0 %
    $ Neuron percentage =  300 / 300 = 100.0 %
    $ Neuron percentage =  99 / 100 = 99.0 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  1193 / 1194 = 99.9162479062 %

    Add L1 regularization

    $ python lenet-300-100.py 0.0001 0 0 0.5 0.01

    A Sample of Summary of Statistics:

    $ sparsity of w_fc1= 10.8520408163 %
    $ sparsity of w_fc2= 18.9233333333 %
    $ sparsity of w_out= 70.8 %
    $ Total Sparsity=  31909 / 266200  =  11.986851991 %
    $ Compression Rate =  8.34247391018
    $ Accuracy without prune: 0.979
    $ Accuracy with prune: 0.9798
    $ Neuron percentage =  572 / 784 = 72.9591836735 %
    $ Neuron percentage =  140 / 300 = 46.6666666667 %
    $ Neuron percentage =  89 / 100 = 89.0 %
    $ Neuron percentage =  10 / 10 = 100.0 %
    $ Total Neuron Percentage =  811 / 1194 = 67.9229480737 %

   You may have the following Summary of statistics:

Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
"""

import tensorflow as tf
from regularizers import *

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np
import sys
from scipy.io import savemat
import matplotlib.pyplot as plt
plt.ion()

lambda_l1 = float(sys.argv[1])
lambda_l2 = float(sys.argv[2])
lambda_dropneuron = float(sys.argv[3])
keep_prob = float(sys.argv[4])  # keep_prob \in (0, 1]
threshold = float(sys.argv[5])

# Parameters
learning_rate_ini = 0.001
training_epochs = 100
batch_size = 256
display_step = 1

# Network Parameters
n_hidden_1 = 300  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def model(_X, _W, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _W['fc1']), _biases['fc1']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['fc2']), _biases['fc2']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_2, keep_prob)
    return tf.matmul(layer_2, _W['out']) + _biases['out']

# Store layers weight & bias
W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

W_prune = W

biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}


def l1(x):
    regularizers = (l1_regularizer(.1)(W['fc1']) + l1_regularizer(.1)(biases['fc1']))
    regularizers += (l1_regularizer(.1)(W['fc2']) + l1_regularizer(.1)(biases['fc2']))
    regularizers += (l1_regularizer(.1)(W['out']) + l1_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def l2(x):
    regularizers = (l2_regularizer(.1)(W['fc1']) + l2_regularizer(.1)(biases['fc1']))
    regularizers += (l2_regularizer(.1)(W['fc2']) + l2_regularizer(.1)(biases['fc2']))
    regularizers += (l1_regularizer(.1)(W['out']) + l1_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def dropneuron(x):
    regularizers = (lo_regularizer(.1)(W['fc1'])) + tf.reduce_mean(li_regularizer(.1)(W['fc1']))
    regularizers += (lo_regularizer(.1)(W['fc2'])) + tf.reduce_mean(li_regularizer(.1)(W['fc2']))
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
    y_prune = y_noprune
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

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_loss += l / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_loss)
    print "Optimization Finished!"

    accuracy_noprune = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    w_fc1_, w_fc1 = prune(W['fc1'])
    W_prune['fc1'] = W['fc1'].assign(w_fc1, use_locking=False)
    print "w_fc1 =", '\n',  w_fc1, "shape = ", np.shape(w_fc1), '\n'
    w_fc2_, w_fc2 = prune(W['fc2'])
    W_prune['fc2'] = W['fc2'].assign(w_fc2, use_locking=False)
    print "w_fc2 =", '\n',  w_fc2, "shape = ", np.shape(w_fc2), '\n'
    w_out_, w_out = prune(W['out'])
    W_prune['out'] = W['out'].assign(w_out, use_locking=False)
    print "w_out =", '\n',  w_out, "shape = ", np.shape(w_out), '\n'
    sess.run(W_prune)

    sparsity = np.count_nonzero(w_fc1)
    sparsity += np.count_nonzero(w_fc2)
    sparsity += np.count_nonzero(w_out)

    print "sparsity of w_fc1=", \
        float(np.count_nonzero(w_fc1))/float(np.size(w_fc1))*100, "%"
    print "sparsity of w_fc2=", \
        float(np.count_nonzero(w_fc2))/float(np.size(w_fc2))*100, "%"
    print "sparsity of w_out=", \
        float(np.count_nonzero(w_out))/float(np.size(w_out))*100, "%"

    num_parameter = np.size(w_fc1)
    num_parameter += np.size(w_fc2)
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
    neuron_left, neuron_total = neuron_layer(w_fc1, w_fc2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_fc2, w_out)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_output(w_out)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    print "Total Neuron Percentage = ", \
        neuron_left_, "/", neuron_total_, "=", float(neuron_left_)/float(neuron_total_)*100, "%"

    savemat('result/result_lenet-300-100.mat',
            {'w_fc1_': w_fc1_,
             'w_fc2_': w_fc2_,
             'w_out_': w_out_,
             'w_fc1': w_fc1,
             'w_fc2': w_fc2,
             'w_out': w_out,
             'learning_rate': learning_rate_ini,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'lambda_dropneuron': lambda_dropneuron,
             'keep_prob': keep_prob,
             'threshold': threshold,
             'accuracy_prune': accuracy_prune,
             'accuracy_noprune': accuracy_noprune})
