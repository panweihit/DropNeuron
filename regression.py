# -*- coding: utf-8 -*-

'''
Implementation of Sparse Regression example using TensorFlow library.
This example is simply following the experimental procedure in compressive sensing
or sparse signal recovery problems

References:
    - E.J. Cande`s and T. Tao. Decoding by linear programming. Information Theory, IEEE Transactions
    on, 51(12):4203–4215, 2005.

    - D.L. Donoho. Compressed sensing. Information Theory, IEEE Transactions on, 52(4):1289–1306, 2006.

================================How to run this script=================================
1. you can run the following command using DropNeuron

    $ python regression.py 50 0 250 1 0.01

 Summary of statistics

    $ maximum of test error is  2.32273391066
    $ normalised mean square error of test error is  0.000357877730384
    $ sparsity of w_fc1= 2.0 %
    $ sparsity of w_out= 20.0 %
    $ Total Sparsity=  3 / 105  =  2.85714285714 %
    $ Compression Rate =  35.0
    $ Neuron percentage =  2 / 20 = 10.0 %
    $ Neuron percentage =  1 / 5 = 20.0 %
    $ Neuron percentage =  1 / 1 = 100.0 %
    $ Total Neuron Percentage =  4 / 26 = 15.3846153846 %


2.you can run the following command using Dropout

    $ python regression.py 20 0 0 0.5 0.01

 Summary of statistics

    $ maximum of test error is  132.45430406
    $ normalised mean square error of test error is  0.539424416323
    $ sparsity of w_fc1= 58.0 %
    $ sparsity of w_out= 100.0 %
    $ total Sparsity=  63 / 105  =  60.0 %
    $ compression Rate =  1.66666666667
    $ neuron percentage =  20 / 20 = 100.0 %
    $ neuron percentage =  5 / 5 = 100.0 %
    $ neuron percentage =  1 / 1 = 100.0 %
    $ Total Neuron Percentage =  26 / 26 = 100.0 %



Author: Wei Pan
Contact: w.pan11@imperial.ac.uk
         dropneuron@gmail.com
'''


import tensorflow as tf
from regularizers import *
# import sklearn
import sys
import numpy as np
from scipy import stats
from scipy.io import savemat
rng = np.random

# Parameters
training_epochs = 100
batch_size = 1
learning_rate_ini = 0.001
lambda_l1 = float(sys.argv[1])
lambda_l2 = float(sys.argv[2])
lambda_dropneuron = float(sys.argv[3])
keep_prob = float(sys.argv[4])  # keep_prob \in (0, 1]
threshold = float(sys.argv[5])

display_step = 1

SEED = 66478  # Set to None for random seed.

###############################################################################
# Generating simulated data with Gaussian weigthts
# np.random.seed(0)
n_samples, n_input = 1000, 20
X = 10*np.random.randn(n_samples, n_input)  # Create Gaussian data
# Create weigts with a precision lambda_.
lambda_ = 0.01
w = np.zeros(n_input)
# Only keep 20*0.1 = 2 nonzeros
perc_sparsity = 0.1
relevant_features = np.random.randint(0, n_input, int(n_input*perc_sparsity))
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))

# Create noise with a precision alpha of 50.
alpha_ = 1.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)

# Create the target
Y = np.dot(X, w) + 0.0*noise

perc_train = 0.5
n_train = int(np.floor(n_samples*perc_train))
print(n_train)
train_X = np.array(X[:n_train, :])
train_Y = np.array(Y[:n_train, ])
test_X = np.array(X[n_train:, :])
test_Y = np.array(Y[n_train:, ])

n_samples = train_X.shape[0]
###############################################################################

# model = LassoCV()
# model.fit(train_X, train_Y)
# print "w_true =", np.asarray(w), '\n'
# print "The model is", model, '\n'
# print "The estimated lasso estimation is", model.coef_, '\n'

###############################################################################


# Network Parameters
n_hidden_1 = 5  # 1st layer num features

input = tf.placeholder("float", [None, n_input])
output = tf.placeholder("float", [None])
W0 = tf.Variable(tf.random_normal([n_input, 1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

'''
You can change the NN architecture here
'''
W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, 1])),
}

W_prune = W

biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([1])),
}

# Building the encoder
def model(x):
    layer_1 = tf.add(tf.matmul(x, W['fc1']), biases['fc1'])
    layer_1 = tf.nn.dropout(layer_1, keep_prob, seed=SEED)
    layer_out = tf.add(tf.matmul(layer_1, W['out']), biases['out'])
    layer_out = tf.nn.dropout(layer_out, keep_prob, seed=SEED)
    return layer_out


def l1(x):
    # L1 regularization for the fully connected parameters.
    regularizers = tf.reduce_mean(l1_regularizer(.1)(W['fc1']) + l1_regularizer(.1)(biases['fc1']))
    regularizers += tf.reduce_mean(l1_regularizer(.1)(W['out']) + l1_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def l2(x):
    # L2 regularization for the fully connected parameters.
    regularizers = tf.reduce_mean(l2_regularizer(.1)(W['fc1']) + l2_regularizer(.1)(biases['fc1']))
    regularizers += tf.reduce_mean(l2_regularizer(.1)(W['out']) + l2_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


def dropneuron(x):
    # DropNeuron regularization for dropping neurons in the fully connected layer.
    regularizers = tf.reduce_mean(lo_regularizer(.1)(W['fc1']))
    regularizers += tf.reduce_mean(li_regularizer(.1)(W['fc1']))
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
pred = model(input)

# Define cost and optimizer
loss_mse = tf.reduce_mean(tf.square(output - pred))
meansquare_test_output = tf.reduce_mean(tf.square(output))
loss_mape = tf.div(loss_mse, meansquare_test_output)
# loss_mape = tf.div(tf.reduce_mean(tf.abs(output - pred)), tf.reduce_mean(tf.abs(output)))
cost = tf.reduce_mean(tf.square(output - pred))
cost += l1(lambda_l1)
cost += l2(lambda_l2)
cost += dropneuron(lambda_dropneuron)


def display():
    print "train mse=", sess.run(loss_mse, feed_dict={input: np.asarray(train_X), output: np.asarray(train_Y)})
    print "test mse=", sess.run(loss_mse, feed_dict={input: np.asarray(test_X), output: np.asarray(test_Y)})
    output_pred = sess.run(pred, feed_dict={input: np.asarray(test_X), output: np.asarray(test_Y)})
    output_pred = np.asarray(output_pred)
    output_pred = output_pred.reshape(1, output_pred.shape[0])[0]
    output_test = test_Y
    test_error = output_test-output_pred
    print "maximum of test error is ", np.max(abs(test_error))
    mse = np.mean(test_error**2)/np.mean(test_Y**2)
    print "normalised mean square error of test error is ", mse
    return mse


# optimizer = tf.train.GradientDescentOptimizer(learning_rate_ini).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate_ini, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate_ini, initial_accumulator_value=0.1, use_locking=False).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate_ini, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)

init = tf.initialize_all_variables()


def gen_batches(data, batch_size):
    """ Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch

    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


with tf.Session() as sess:
    sess.run(init)

    print "X shape = ", np.shape(np.asarray(train_X)), '\n'
    print "Y shape = ", np.shape(np.asarray(train_Y)), '\n'

    shuff = zip(np.asarray(train_X), np.asarray(train_Y))
    # Fit all training data
    for epoch in range(training_epochs):
        # np.random.shuffle(shuff)
        batches = [_ for _ in gen_batches(shuff, batch_size)]

        for batch in batches:
            x0, y0 = zip(*batch)
            x0 = np.reshape(x0, (batch_size, n_input))
            y0 = np.reshape(y0, (batch_size, ))
            sess.run(optimizer, feed_dict={input: x0, output: y0})

        print "epoch:", epoch, "Train mse=", sess.run(loss_mse, feed_dict={input: x0, output: y0})
        print "epoch:", epoch, "Train cost=", sess.run(cost, feed_dict={input: x0, output: y0})

    mse_noprune = display()

    w_fc1 = sess.run(W['fc1'])
    w_fc1 = np.asarray(w_fc1)
    print "w_fc1 without prune =", '\n',  w_fc1, "shape = ", np.shape(W['fc1']), '\n'
    w_out = sess.run(W['out'])
    w_out = np.asarray(w_out)
    print "w_out without prune =", '\n',  w_out, "shape = ", np.shape(W['out']), '\n'

    w_fc1_, w_fc1 = prune(W['fc1'])
    W_prune['fc1'] = W['fc1'].assign(w_fc1, use_locking=False)
    print "w_fc1 with prune = ", '\n',  w_fc1, "shape = ", np.shape(W['fc1']), '\n'
    w_out_, w_out = prune(W['out'])
    W_prune['out'] = W['out'].assign(w_out, use_locking=False)
    print "w_out with prune =", '\n',  w_out, "shape = ", np.shape(W['out']), '\n'
    sess.run(W_prune)

    mse_prune = display()
    
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

    savemat('result/result_regression.mat',
            {'w_fc1_': w_fc1_,
             'w_out_': w_out_,
             'w_fc1': w_fc1,
             'w_out': w_out,
             'true_solution': w,
             'learning_rate': learning_rate_ini,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'lambda_dropneuron': lambda_dropneuron,
             'keep_prob': keep_prob,
             'threshold': threshold,
             'mse_noprune': mse_noprune,
             'mse_prune': mse_prune})
