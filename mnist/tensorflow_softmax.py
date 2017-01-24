from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

## A lot like theano, we describe things symbolically first
## then compile and compute.
x = tf.placeholder(tf.float32, [None, 784])
## This is a placeholder value of shape (None, 784), where none can be any 
## value

## Model variables/parameters are then initialized as
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
## where the weights and biases are variables since they're
## expected to be modified throughout the computation.

## the model is then 1 line
y = tf.nn.softmax(tf.matmul(x, W) + b)


## Implement the cost function
## place holder for true values
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

## Training is done via efficiently implemented back prop
## since tf knows the complete network structure
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

## Now compile the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

## Now train
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

## testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) ## returns bools
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) ## recast and average
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



