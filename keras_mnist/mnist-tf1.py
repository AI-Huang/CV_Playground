#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 15:21:16
# @Author  : Your Name (you@example.org)
# @Link    : https://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# @Version : $Id$
# @Description: MNIST机器学习入门

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    print("Tensorflow version: ", tf.__version__)
    # Load data
    mnist = input_data.read_data_sets(
        train_dir="D:\\DeepLearningData\\datasets\\MNIST\\tf", one_hot=True)
    # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
    # print(type(mnist))
    print(mnist.train.images.shape)  # (55000, 784)
    print(mnist.train.labels.shape)  # (55000, 10)
    print(mnist.test.labels.shape)  # (10000, 10)

    # 建立回归模型
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # 训练模型
    for i in range(1000):
        # print(f"{i}")
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 评估模型


if __name__ == "__main__":
    main()
