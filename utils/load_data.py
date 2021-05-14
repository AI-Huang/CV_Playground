#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-06-20 02:47
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import os
import numpy as np
import tensorflow as tf


def load_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    print(y_train.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    num_classes = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


def load_data(dataset="mnist", if_categorical=True, if_expand_dims=False, if_normalized=False):
    if dataset == "mnist":
        # Load mnist data
        dataset = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images,
                                       test_labels) = dataset.load_data()

        if if_categorical:
            # num_classes = np.max(train_labels) + 1  # 10 classes for mnist
            train_labels = tf.keras.utils.to_categorical(
                train_labels)  # to one-hot
            test_labels = tf.keras.utils.to_categorical(
                test_labels)  # to one-hot

        if if_expand_dims:
            train_images = np.expand_dims(train_images, -1)
            test_images = np.expand_dims(test_images, -1)

        if if_normalized:
            train_images, test_images = train_images / 255.0, test_images / 255.0

        return (train_images, train_labels), (test_images,
                                              test_labels)


def main():
    load_mnist()


if __name__ == "__main__":
    main()
