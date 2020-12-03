#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-20 21:54
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D,  Flatten
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import GlobalAveragePooling2D


available_attention = ["official", "senet"]


def senet_attention(inputs):
    """A SeNetAttention block implementation with Keras functional API
    Input:
        inputs: 2D feature maps with shape (H, W, C)
    """
    num_channels = inputs.shape[-1]
    x = GlobalAveragePooling2D()(inputs)
    x = tf.expand_dims(x, -2)
    x = tf.expand_dims(x, -2)  # make output shape be (None, 1, 1, C)
    x = Dense(num_channels, activation="relu")(x)
    # The last FC layer generates the scale (or query) tensor
    x = Dense(num_channels, activation="sigmoid")(x)

    return x * inputs  # multiply (None, 1, 1, C) and (None, H, W, C)


def AttentionLeNet5(input_shape, num_classes=None, attention=None):
    """AttentionLeNet-5 network built with Keras
    Inputs:
        input_shape: input shape of the element of the batched data, e.g., (32, 32, 1), (28, 28, 1)
        attention: attention type, one of ["official", "senet"], default None.
    """
    input_ = Input(shape=input_shape)

    model_head = Sequential()
    model_head.add(input_)
    model_head.add(Conv2D(filters=6, kernel_size=(5, 5),
                          padding="valid", activation="relu"))
    model_head.add(MaxPool2D(strides=2))
    model_head.add(Conv2D(filters=16, kernel_size=(5, 5),
                          padding="valid", activation="relu"))
    model_head.add(MaxPool2D(strides=2))

    model_top = Sequential()
    model_top.add(Flatten())
    model_top.add(Dense(120, activation="relu"))
    model_top.add(Dense(84, activation="relu"))
    model_top.add(Dense(10, activation='softmax'))

    # Add Attention afer Flatten Layer
    if attention is not None:
        if not attention in available_attention:
            raise ValueError(
                f"""attention argument must be in ["official", "senet"]""")
    if attention == "official":
        x_attention = Attention(model_head.output)
    elif attention == "senet":
        x_attention = senet_attention(model_head.output)

    x_attention = model_top(x_attention)
    model = Model(inputs=input_, outputs=x_attention)

    # model.build()

    return model


def AttentionLeNet5_test():
    num_classes = 10
    input_shape = (28, 28, 1)
    model = AttentionLeNet5(input_shape=input_shape,
                            num_classes=num_classes, attention="senet")


def main():
    AttentionLeNet5_test()


if __name__ == "__main__":
    main()
