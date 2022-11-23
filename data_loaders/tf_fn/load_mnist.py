#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-06-20 02:47
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np
import tensorflow as tf


def load_data(dataset="mnist", if_categorical=True, if_expand_dims=False, if_normalized=False):
    if dataset == "mnist":
        # Load mnist data
        dataset = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) \
            = dataset.load_data()

    if if_categorical:
        train_labels = tf.keras.utils.to_categorical(
            train_labels)
        test_labels = tf.keras.utils.to_categorical(
            test_labels)

    if if_expand_dims:
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

    if if_normalized:
        train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), \
        (test_images, test_labels)


def main():
    dataset = "mnist"
    (train_images, train_labels), (test_images, test_labels) = \
        load_data(dataset=dataset,
                  if_categorical=True,
                  if_expand_dims=True,
                  if_normalized=False)


if __name__ == "__main__":
    main()
