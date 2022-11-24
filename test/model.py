#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-23-22 18:39
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)



import tensorflow as tf
import sys
sys.path.append(".")


def main():
    from models.tf_fn.model_utils import create_model_cifar10
    model = create_model_cifar10(input_shape=(32, 32, 3),
                                 depth=20, se_net=True, version=1)
    tf.keras.utils.plot_model(
        model,
        to_file="fig/model.png",
        show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )


if __name__ == "__main__":
    main()
