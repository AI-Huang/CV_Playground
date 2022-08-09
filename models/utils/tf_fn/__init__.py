#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-27-20 21:46
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""Model configuration functions
- create_model
- create_optimizer
- set_metrics

Requirements:
    Kan HUANG's models repo: https://github.com/AI-Huang/models
"""
import tensorflow as tf
from models.keras_fn.lenet import LeNet5, LeCunLeNet5
# Fault ResNet
from models.keras_fn.fault_resnet import resnet_v1, resnet_v2, lr_schedule


def create_model(name, **kwargs):
    """Create model with model's name
    """
    assert "input_shape" in kwargs
    assert "num_classes" in kwargs
    input_shape = kwargs["input_shape"]
    num_classes = kwargs["num_classes"]

    if name == "LeNet5":
        model = LeNet5(input_shape=input_shape, num_classes=num_classes)

    elif name == "LeCunLeNet5":
        model = LeCunLeNet5(input_shape=input_shape, num_classes=num_classes)

    elif name.startswith("AttentionLeNet5"):
        from .attention_lenet import AttentionLeNet5
        model = AttentionLeNet5(input_shape=input_shape,
                                num_classes=num_classes,
                                attention="senet")

    elif name == "ResNet18":
        from models.keras_fn.resnet_extension import ResNet18
        model = ResNet18(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif name == "ResNet34":
        from models.keras_fn.resnet_extension import ResNet34
        model = ResNet34(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif name == "ResNet50":
        from tensorflow.keras.applications.resnet import ResNet50
        model = ResNet50(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif name == "ResNet101":
        from tensorflow.keras.applications.resnet import ResNet101
        model = ResNet101(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif name == "ResNet152":
        from tensorflow.keras.applications.resnet import ResNet152
        model = ResNet152(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif name == "ResNet20v2":  # "ResNet20v2",  "ResNet56v2"
        # hparams: n, version, input_shape, num_classes
        assert "n" in kwargs
        assert "version" in kwargs
        n = kwargs["n"]
        version = kwargs["version"]

        from .fault_resnet import model_depth, resnet_v2, lr_schedule
        depth = model_depth(n=2, version=2)
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
        # TODO
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))

    else:
        raise Exception('Unknown model: ' + name)

    return model


def create_optimizer(name="Adam", **kwargs):

    # Default values
    learning_rate = 0.001

    if "learning_rate" in kwargs:
        learning_rate = kwargs["learning_rate"]
    if name == "Adam":
        optimizer = tf.keras.optimizers.Adam()
    else:
        raise Exception('Unknown optimizer: ' + name)

    return optimizer


def set_metrics():
    """set_metrics, 
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
