#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 15:02
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np
import tensorflow as tf
from data_loaders.tf_fn.augmentations import to_tensor, pad_and_crop
from data_loaders.tf_fn.data_sequences import CIFAR10Sequence


def color_normalize(train_images, test_images):
    mean = [np.mean(train_images[:, :, :, i])
            for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i])
           for i in range(3)]  # [62.9932, 62.0887, 66.7048]
    for i in range(3):
        train_images[:, :, :, i] = (
            train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    return train_images, test_images


def load_cifar10_sequence(**kwargs):
    to_categorical = kwargs["to_categorical"] if "to_categorical" in kwargs else False
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
    shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False
    seed = kwargs["seed"] if "seed" in kwargs else 42
    validation_split = kwargs["validation_split"] if "validation_split" in kwargs else 0
    norm = kwargs["norm"] if "norm" in kwargs else False

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = color_normalize(x_train, x_test)

    to_categorical = True
    if to_categorical:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    # transforms = [to_tensor, pad_and_crop]
    transforms = [pad_and_crop]

    if norm:
        if float(tf.__version__[:3]) <= 2.3:
            raise EnvironmentError(
                f"TensorFLow version {tf.__version__} doesn't support Normalization.")
        try:
            from tensorflow.keras.layers import Normalization
        except:
            from tensorflow.keras.layers.experimental.preprocessing import Normalization

        print("Apply data normalization.")
        transforms.append(
            Normalization(
                mean=(0.49139968, 0.48215827, 0.44653124),
                variance=(0.24703233, 0.24348505, 0.26158768))
        )

    cifar10_sequence_train = CIFAR10Sequence(x_train, y_train,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             seed=seed,
                                             subset="training",
                                             validation_split=validation_split,
                                             transforms=transforms)

    cifar10_sequence_val = CIFAR10Sequence(x_train, y_train,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           subset="validation",
                                           validation_split=validation_split,
                                           transforms=transforms)

    cifar10_sequence_test = CIFAR10Sequence(x_test, y_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            subset="validation",
                                            validation_split=1.0,
                                            transforms=None)

    return cifar10_sequence_train, cifar10_sequence_val, cifar10_sequence_test


def main():
    cifar10_sequence_train, cifar10_sequence_val, cifar10_sequence_test = load_cifar10_sequence()


if __name__ == "__main__":
    main()
