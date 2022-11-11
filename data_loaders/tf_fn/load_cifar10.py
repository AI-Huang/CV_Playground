#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 15:02
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from data_loaders.tf_fn.augmentations import to_tensor, pad_and_crop
from data_loaders.tf_fn.data_sequences import CIFAR10Sequence


def load_cifar10_sequence(**kwargs):
    to_categorical = kwargs["to_categorical"] if "to_categorical" in kwargs else False
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
    shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False
    seed = kwargs["seed"] if "seed" in kwargs else 42
    validation_split = kwargs["validation_split"] if "validation_split" in kwargs else 0

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    to_categorical = True
    if to_categorical:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    cifar10_sequence_train = CIFAR10Sequence(x_train, y_train,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             seed=seed,
                                             subset="training",
                                             validation_split=validation_split,
                                             transforms=[to_tensor, pad_and_crop])
    cifar10_sequence_val = CIFAR10Sequence(x_train, y_train,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           subset="validation",
                                           validation_split=validation_split,
                                           transforms=[to_tensor, pad_and_crop])
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
