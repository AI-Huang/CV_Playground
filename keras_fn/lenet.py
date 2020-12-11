#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-08-20 21:49
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Activation, Flatten


def LeNet5(input_shape, num_classes):
    """LeNet-5 network built with Keras
    Inputs:
        input_shape: input shape of the element of the batched data, e.g., (32, 32, 1), (28, 28, 1).
        num_classes: number of top classifiers, e.g., 2, 10.
        attention: attention type, one of["official", "senet"], default None.
    """
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     padding="valid", activation="relu"))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     padding="valid", activation="relu"))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(10, activation='softmax'))

    model.build()

    return model


def LeCunLeNet5(input_shape, num_classes):
    """LeCunLeNet-5 network built with Keras
    As for LeCun's origin LeNet-5, the activation function is after the pooling layer, and the activation function used is sigmoid.
    Inputs:
        input_shape: input shape of the element of the batched data, e.g., (32, 32, 1), (28, 28, 1).
        num_classes: number of top classifiers, e.g., 2, 10.
        attention: attention type, one of["official", "senet"], default None.
    """
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding="valid"))
    model.add(MaxPool2D(strides=2))
    model.add(Activation("sigmoid"))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding="valid"))
    model.add(MaxPool2D(strides=2))
    model.add(Activation("sigmoid"))
    model.add(Flatten())
    model.add(Dense(120, activation="sigmoid"))
    model.add(Dense(84, activation="sigmoid"))
    model.add(Dense(10, activation='softmax'))

    model.build()

    return model


def LeNet5_test():
    num_classes = 10
    input_shape = (28, 28, 1)
    model = LeCunLeNet5(input_shape=input_shape, num_classes=num_classes)


def main():
    LeNet5_test()


if __name__ == "__main__":
    main()
