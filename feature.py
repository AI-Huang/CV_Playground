#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-05-21 23:08
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


def main():
    dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    num_classes = np.max(train_labels) + 1  # 10 classes
    train_labels = tf.keras.utils.to_categorical(train_labels)  # to one-hot
    test_labels = tf.keras.utils.to_categorical(test_labels)  # to one-hot

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    input_shape = train_images.shape[1:]

    # Model
    input_shape = train_images.shape[1:]
    num_classes = 10
    from models.keras_fn.lenet import LeNet5
    model = LeNet5(input_shape=input_shape, num_classes=num_classes)
    model.summary()


if __name__ == "__main__":
    main()
