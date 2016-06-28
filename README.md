# DropNeuron: Simplifying the Structure of Deep Neural Networks

This is a demo of [DropNeuron](http://arxiv.org/abs/1606.07326). 
We perform various supervised and unsupervised learning tasks in Deep Learning. 
During training, many neurons are dropped which yields a much smaller model size but 
no accuracy lost.

DropNeuron is aimed to train a small model from a large random initialized model, rather than 
compress or reduce a large trained model. DropNeuron can be mixed used with other 
regularization techniques, e.g. Dropout, L1, L2.


## Related Paper
[DropNeuron: Simplifying the Structure of Deep Neural Networks](http://arxiv.org/abs/1606.07326)


If you find DropNeuron useful in your research, please consider citing the paper:

	@inproceedings{pan2016dropneuron,
	  title={DropNeuron: Simplifying the Structure of Deep Neural Networks},
	  author={Pan, Wei and Dong, Hao and Guo, Yike},
	  journal={arXiv preprint arXiv:1606.07326},
	  year={2016}
	}
	


## Usage:

### Installation

**TensorFlow Installation**

The codes requires Tensorflow (version = 0.9) to be installed: 
*[Tensorflow installation instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)*.


**Using New [Regularizers](regularizers.py)**

The key file is [regularizers.py](regularizers.py) which is implemented based on the official [TensorFlow](https://github.com/tensorflow/tensorflow/) implementation: 
[regularizers.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py).
The difference is that two new regularizes are added: *lo_regularizer* and *li_regularizer* to 
regularize the outgoing and incoming connections of neurons.
see the [paper]((http://arxiv.org/abs/1606.07326)) for more details.  


One option is substitute the offical [regularizers.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py). 
with the new [regularizers.py](regularizers.py). Typically, we use the following command on both Linux and Mac (I'm using Ubuntu 14.04 and MAC OS 10.11.4)

    pip show tensorflow
    cd /usr/local/lib/python2.7/site-packages/tensorflow/contrib/layers/python/layers


The other option is import 'regularizers' in the header of the file
```python
from regularizers import *
```


### Run in terminal

Look at the instructions and a sample of results on top of the script of each example.

You will use a command looking like this with FIVE input parameters. You can use the parameter in the examples.


    python examplename.py argv[1] argv[2] argv[3] argv[4] argv[5]


| Input| Description |
|-----|-----|
| argv[1] | L1 regularization parameter |
| argv[2] | L2 regularization parameter |
| argv[3] | Dropout keep probability |
| argv[4] | DropNeuron parameter |
| argv[5] | pruning threshold |



### Model and Problem Formulation:

A typical model and cost function specification are as follows. This is an example on LeNet-300-100

```python

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def model(_X, _W, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _W['h1']), _biases['b1'])) #Hidden layer with RELU activation
    tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['h2']), _biases['b2'])) #Hidden layer with RELU activation
    tf.nn.dropout(layer_2, keep_prob)
    return tf.matmul(layer_2, _W['out']) + _biases['out']

# Store layers weight & bias
W = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}

def dropneuron(x):
    regularizers = (lo_regularizer(.1)(W['h1'])) + tf.reduce_mean(li_regularizer(.1)(W['h1']))
    regularizers += (lo_regularizer(.1)(W['h2'])) + tf.reduce_mean(li_regularizer(.1)(W['h2']))
    regularizers += (lo_regularizer(.1)(W['out'])) + tf.reduce_mean(li_regularizer(.1)(W['out']))
    regularizers = x * regularizers
    return regularizers
    
# Construct model
pred = model(x, W, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
cost = loss
cost += dropneuron(0.001)
    
```


### Using High Level API: 

We use [TFLean](http://tflearn.org) to implement AlexNet. Similarly, you may check [Keras](http://keras.io/) as an alternative.

```python

# Building 'AlexNet'
# Placeholders for data and labels
X = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
Y = tf.placeholder(shape=(None, n_class), dtype=tf.float32)
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
loss += 0.001 * regularizers

```
 
   
## Examples:

**[Sparse Regression](regression.py)** 

Implement sparse regression using a fully connected network with one hidden layer. 
This example is synthetic and following the standard setup in Compressive Sensing and Sparse Signal Recovery papers 

Interestingly, we apply DropNeuron to recover the exact solution with linear activation function! 
Check papers of Emmanuel Candes, Terrence Tao and David Donoho on compressive sensing and results on performance guarantee.

**[Autoencoder](autoencoder.py)**

Implement autoencoder for feature extraction of MNIST dataset.

**[LeNet-300-100](lenet-300-100.py)** 

Implement LeNet for classification of MNIST dataset. 
LeNet-300-100 is a fully connected network with two hidden layers, with 300 and 100 neurons each.

**[LeNet-5](lenet-5.py)**

Implement LeNet for classification of MNIST dataset. 
LeNet-5 is a convolutional network that has two convolutional layers and two fully connected layers

**[ConvNet](convnet.py)**

This is a modification of the official TensorFlow tutorial 
on ['convolutional.py'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
). Check the regularizers specification after model specification in the code.

**[AlexNet](alexnet.py)** 

This is a modification of the TFLearn example ['Alexnet.py'](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py).
There are more [examples](https://github.com/tflearn/tflearn/tree/master/examples) 
implemented using TFLearn by [Aymeric Damien](https://github.com/aymericdamien).
You can apply DropNeuron to more complicated examples. 


