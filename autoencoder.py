# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

================================How to run this script=================================

1. you can run the following command using DropNeuron

    $ python autoencoder.py 0.00001  0 0.00001 1  0.005


   You may have the following Summary of statistics:

    $ sparsity of w_encoder_h1= 16.0036670918 %
    $ sparsity of w_encoder_h2= 44.4702148438 %
    $ sparsity of w_decoder_h1= 54.1137695312 %
    $ sparsity of w_decoder_h2= 18.141143176 %
    $ Total Sparsity=  42341 / 217088  =  19.5040720814 %
    $ Compression Rate =  5.12713445596
    $ NMSE without prune: 0.0121042
    $ NMSE with prune: 0.0121184
    $ Neuron percentage =  420 / 784 = 53.5714285714 %
    $ Neuron percentage =  127 / 128 = 99.21875 %
    $ Neuron percentage =  61 / 64 = 95.3125 %
    $ Neuron percentage =  121 / 128 = 94.53125 %
    $ Neuron percentage =  629 / 784 = 80.2295918367 %
    $ Total Neuron Percentage =  1358 / 1888 = 71.9279661017 %



2. you can run the following command without Regularisation or DropOut

    $ python autoencoder.py 0.00001  0 0  0.5  0.005

    You may have the following Summary of statistics:

    $ sparsity of w_encoder_h1= 15.1845503827 %
    $ sparsity of w_encoder_h2= 46.2890625 %
    $ sparsity of w_decoder_h1= 52.5268554688 %
    $ sparsity of w_decoder_h2= 17.5392617985 %
    $ Total Sparsity=  40934 / 217088  =  18.8559478184 %
    $ Compression Rate =  5.30336639468
    $ NMSE without prune: 0.0115187
    $ NMSE with prune: 0.0115109
    $ Neuron percentage =  459 / 784 = 58.5459183673 %
    $ Neuron percentage =  127 / 128 = 99.21875 %
    $ Neuron percentage =  62 / 64 = 96.875 %
    $ Neuron percentage =  121 / 128 = 94.53125 %
    $ Neuron percentage =  784 / 784 = 100.0 %
    $ Total Neuron Percentage =  1553 / 1888 = 82.2563559322 %




    $ python autoencoder.py 0  0  0  0.5  0.005

    You may have the following Summary of statistics:

    $ sparsity of w_encoder_h1= 99.5744977679 %
    $ sparsity of w_encoder_h2= 99.3896484375 %
    $ sparsity of w_decoder_h1= 99.4506835938 %
    $ sparsity of w_decoder_h2= 99.6083785077 %
    $ Total Sparsity=  216173 / 217088  =  99.5785119399 %
    $ Compression Rate =  1.00423272102
    $ NMSE without prune: 0.0311838
    $ NMSE with prune: 0.0311848
    $ Neuron percentage =  784 / 784 = 100.0 %
    $ Neuron percentage =  128 / 128 = 100.0 %
    $ Neuron percentage =  64 / 64 = 100.0 %
    $ Neuron percentage =  128 / 128 = 100.0 %
    $ Neuron percentage =  784 / 784 = 100.0 %
    $ Total Neuron Percentage =  1888 / 1888 = 100.0 %



Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
"""

import tensorflow as tf
from regularizers import *
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

learning_rate_ini = 0.001
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 128  # 1st layer num features
n_hidden_2 = 64  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
input = tf.placeholder("float", [None, n_input])

W = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=1)),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=1)),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev=1)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=1)),
}

W_prune = W

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=1)),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=1)),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=1)),
    'decoder_b2': tf.Variable(tf.random_normal([n_input], stddev=1)),
}


# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['encoder_h1']), biases['encoder_b1']))
    tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['encoder_h2']), biases['encoder_b2']))
    tf.nn.dropout(layer_2, keep_prob)
    return layer_2


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['decoder_h1']), biases['decoder_b1']))
    tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['decoder_h2']), biases['decoder_b2']))
    tf.nn.dropout(layer_2, keep_prob)
    return layer_2


def l1(x):
    regularizers = (l1_regularizer(.1)(W['encoder_h1']) + l1_regularizer(.1)(biases['encoder_b1']))
    regularizers += (l1_regularizer(.1)(W['encoder_h2']) + l1_regularizer(.1)(biases['encoder_b2']))
    regularizers += (l1_regularizer(.1)(W['decoder_h1']) + l1_regularizer(.1)(biases['decoder_b1']))
    regularizers += (l1_regularizer(.1)(W['decoder_h2']) + l1_regularizer(.1)(biases['decoder_b2']))
    regularizers = x * regularizers
    return regularizers


def l2(x):
    regularizers = (l2_regularizer(.1)(W['encoder_h1']) + l2_regularizer(.1)(biases['encoder_b1']))
    regularizers += (l2_regularizer(.1)(W['encoder_h2']) + l2_regularizer(.1)(biases['encoder_b2']))
    regularizers += (l2_regularizer(.1)(W['decoder_h1']) + l2_regularizer(.1)(biases['decoder_b1']))
    regularizers += (l2_regularizer(.1)(W['decoder_h2']) + l2_regularizer(.1)(biases['decoder_b2']))
    regularizers = x * regularizers
    return regularizers


def dropneuron(x):
    regularizers = (lo_regularizer(.1)(W['encoder_h1'])) + tf.reduce_mean(li_regularizer(.1)(W['encoder_h1']))
    regularizers += (lo_regularizer(.1)(W['encoder_h2'])) + tf.reduce_mean(li_regularizer(.1)(W['encoder_h2']))
    regularizers += (lo_regularizer(.1)(W['decoder_h1'])) + tf.reduce_mean(li_regularizer(.1)(W['decoder_h1']))
    regularizers += (lo_regularizer(.1)(W['decoder_h2'])) + tf.reduce_mean(li_regularizer(.1)(W['decoder_h2']))
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
encoder_op = encoder(input)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = input
# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost = loss
cost += l1(lambda_l1)
cost += l2(lambda_l2)
cost += dropneuron(lambda_dropneuron)

optimizer = tf.train.AdamOptimizer(learning_rate_ini, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, cost_value = sess.run([optimizer, cost], feed_dict={input: batch_xs})
            loss_value = sess.run(loss, feed_dict={input: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(loss_value)
    print("Optimization Finished!")

    nmse_noprune = sess.run(loss, feed_dict={input: mnist.test.images})

    w_encoder_h1_, w_encoder_h1 = prune(W['encoder_h1'])
    W_prune['encoder_h1'] = W['encoder_h1'].assign(w_encoder_h1, use_locking=False)
    print "w_encoder_h1 =", '\n',  w_encoder_h1, "shape = ", np.shape(w_encoder_h1)
    w_encoder_h2_, w_encoder_h2 = prune(W['encoder_h2'])
    W_prune['encoder_h2'] = W['encoder_h2'].assign(w_encoder_h2, use_locking=False)
    print "w_encoder_h2 =", '\n',  w_encoder_h2, "shape = ", np.shape(w_encoder_h2)
    w_decoder_h1_, w_decoder_h1 = prune(W['decoder_h1'])
    W_prune['decoder_h1'] = W['decoder_h1'].assign(w_decoder_h1, use_locking=False)
    print "w_decoder_h1 =", '\n',  w_decoder_h1, "shape = ", np.shape(w_decoder_h1)
    w_decoder_h2_, w_decoder_h2 = prune(W['decoder_h2'])
    W_prune['decoder_h2'] = W['decoder_h2'].assign(w_decoder_h2, use_locking=False)
    print "w_decoder_h2 =", '\n',  w_decoder_h2, "shape = ", np.shape(w_decoder_h2)
    sess.run(W_prune)

    sparsity = np.count_nonzero(w_encoder_h1)
    sparsity += np.count_nonzero(w_encoder_h2)
    sparsity += np.count_nonzero(w_decoder_h1)
    sparsity += np.count_nonzero(w_decoder_h2)

    print "sparsity of w_encoder_h1=", \
        float(np.count_nonzero(w_encoder_h1))/float(np.size(w_encoder_h1))*100, "%"
    print "sparsity of w_encoder_h2=", \
        float(np.count_nonzero(w_encoder_h2))/float(np.size(w_encoder_h2))*100, "%"
    print "sparsity of w_decoder_h1=", \
        float(np.count_nonzero(w_decoder_h1))/float(np.size(w_decoder_h1))*100, "%"
    print "sparsity of w_decoder_h2=", \
        float(np.count_nonzero(w_decoder_h2))/float(np.size(w_decoder_h2))*100, "%"

    num_parameter = np.size(w_encoder_h1)
    num_parameter += np.size(w_encoder_h2)
    num_parameter += np.size(w_decoder_h1)
    num_parameter += np.size(w_decoder_h2)
    total_sparsity = float(sparsity)/float(num_parameter)

    print "Total Sparsity= ", sparsity, "/", num_parameter, \
        " = ", total_sparsity*100, "%"
    print "Compression Rate = ", float(num_parameter)/float(sparsity)

    nmse_prune = sess.run(loss, feed_dict={input: mnist.test.images})

    print "NMSE without prune:", nmse_noprune
    print "NMSE with prune:", nmse_prune

    # Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={input: mnist.test.images[:examples_to_show]})

    neuron_left_ = 0
    neuron_total_ = 0
    neuron_left, neuron_total = neuron_input(w_encoder_h1)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_encoder_h1, w_encoder_h2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_encoder_h2, w_decoder_h1)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_layer(w_decoder_h1, w_decoder_h2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    neuron_left, neuron_total = neuron_output(w_decoder_h2)
    neuron_left_ += neuron_left
    neuron_total_ += neuron_total
    print "Total Neuron Percentage = ", \
        neuron_left_, "/", neuron_total_, "=", float(neuron_left_)/float(neuron_total_)*100, "%"


    savemat('result/result_autoencoder_dropout_only.mat',
            {'w_encoder_h1_': w_encoder_h1_,
             'w_encoder_h2_': w_encoder_h2_,
             'w_decoder_h1_': w_decoder_h1_,
             'w_decoder_h2_': w_decoder_h2_,
             'w_encoder_h1': w_encoder_h1,
             'w_encoder_h2': w_encoder_h2,
             'w_decoder_h1': w_decoder_h1,
             'w_decoder_h2': w_decoder_h2,
             'learning_rate': learning_rate_ini,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'lambda_dropneuron': lambda_dropneuron,
             'keep_prob': keep_prob,
             'threshold': threshold,
             'accuracy_prune': nmse_prune,
             'accuracy_noprune': nmse_noprune})

    # Compare original images with their reconstructions
    f1, a1 = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(examples_to_show):
        # a1[i].imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='Greys_r')
        a1[i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a1[i].axis('off')
    f1.show()
    plt.savefig('m1.pdf', format='pdf')
    f2, a2 = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(examples_to_show):
        # a2[i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='Greys_r')
        a2[i].imshow(np.reshape(encode_decode[i], (28, 28)))
        a2[i].axis('off')
    f2.show()
    plt.savefig('m2.5.pdf', format='pdf')
    plt.draw()
    plt.waitforbuttonpress()

