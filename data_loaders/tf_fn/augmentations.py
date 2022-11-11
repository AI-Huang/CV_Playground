#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 14:59
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from tensorflow.keras import layers

minor_version = int(tf.__version__.split('.')[1])
if minor_version <= 3:
    # TensorFlow v2.3
    from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomCrop
else:
    from tensorflow.keras.layers import RandomFlip, RandomCrop


def to_tensor_img(x):
    return x / 255


def to_tensor():
    return layers.Lambda(lambda x: to_tensor_img(x))


to_tensor = to_tensor()
to_tensor = tf.keras.Sequential([
    to_tensor
])

pad_and_crop = tf.keras.Sequential([
    layers.ZeroPadding2D(padding=(4, 4)),
    RandomFlip("horizontal"),
    RandomCrop(32, 32)
])
