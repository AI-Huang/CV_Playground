#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-06-20 02:47
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    :

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import tensorflow as tf
from LeNet import LeNet5


def main():
    print(f"Tensorflow version: {tf.__version__}")

    # Load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # train the model
    epochs = 1
    with tf.device('/device:GPU:0'):
        num_classes = 10
        input_shape = (28, 28, 1)
        model = LeNet5(input_shape=input_shape, num_classes=num_classes)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=epochs)
        model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()
