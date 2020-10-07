# Jessica Chen
# 10/07/2020
# BMW Lab

# Basic TensorFlow tutorial
# Source: https://adventuresinmachinelearning.com/python-tensorflow-tutorial/

import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf

# basic introduction
"""
# create tf constant
const = tf.constant(2.0, name="const")

# create tf variables
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

# create tf operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# set up variable initialization
init_op = tf.global_variables_initializer()

# start tf session object for building static graphs
with tf.Session() as sess:
    # initialize variables
    sess.run(init_op)
    # compute output of graph
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))

"""

# load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# trying to fix mnist tutorial modulenotfound error
"""
import tensorflow_datasets as tfds
# Construct a tf.data.Dataset
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
"""


# python optimization variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare training data placeholders
# input x: for 28x28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# output data placeholder: 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# declare weights connecting input to hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# declare weights connecting hidden layer to output
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate output of hidden layer
# multiply weights W1 by inputs x, then add bias b1
hidden_out = tf.add(tf.matmul(x, W1), b1)
# apply rectified linear unit activation function tf.nn.relu
hidden_out = tf.nn.relu(hidden_out)

# calculate hidden layer output
# multiply weights, add bias, then apply softmax activation function tf.nn.softmax
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# cross entropy cost function
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# set up initialization operator
init_op = tf.global_variables_initializer()

# define accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# to find reduce_mean, must cast correct_prediction boolean to a float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))