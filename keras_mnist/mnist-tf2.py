#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-06-20 02:47
# @Author  : Your Name (you@example.org)
# @Link    : https://www.tensorflow.org/tutorials/quickstart/beginner

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    print("Tensorflow version: ", tf.__version__)
    # Load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(
        path=os.path.join("D:\\DeepLearningData\\datasets", "mnist.npz"))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 模型构建
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()
