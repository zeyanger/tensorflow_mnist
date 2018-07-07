#！/usr/bin/env.python
# _*_ coding:utf-8 _*_


import os
import tensorflow as tf

import model
import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)

# model
with tf.variable_scope('convolution'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolution(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10], name='y')
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = -tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存
saver = tf.train.Saver()

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summay_writer = tf.summary.FileWriter('/tmp/mnist_log/1', sess.graph)
    summay_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
