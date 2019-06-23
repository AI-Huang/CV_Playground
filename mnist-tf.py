#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 15:21:16
# @Author  : Your Name (you@example.org)
# @Link    : https://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# @Version : $Id$
# @Description: MNIST机器学习入门

import numpy
import matplotlib.pyplot as plt
import os
import platform
import tensorflow as tf


def main():
    print("Python version:")
    print(platform.python_version())
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 模型构建
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # 训练方法构建
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 10000个数据，100次迭代一边，1000次只有10个epoch
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            print("epoch %d ends" % (i//100))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    main()
