#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-05-21 20:56
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)


import tensorflow as tf
import sys
sys.path.append(".")


def main():
    from models.tf_fn.model_utils import create_model_cifar10
    model = create_model_cifar10(input_shape=(32, 32, 3),
                                 depth=20, se_net=True, version=1)


if __name__ == "__main__":
    main()
