#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-22-20 17:39
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)


"""attention_visualization
Visualize the Attention layer.
"""
import os


def main():
    # Prepare data
    dataset_name = "mnist"

    if dataset_name == "mnist":
        dataset = tf.keras.datasets.mnist
        (train_images, train_labels), \
            (test_images, test_labels) = dataset.load_data()

        num_classes = np.max(train_labels) + 1  # 10 classes
        train_labels = tf.keras.utils.to_categorical(
            train_labels)  # to one-hot
        test_labels = tf.keras.utils.to_categorical(
            test_labels)  # to one-hot

        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)
        input_shape = train_images.shape[1:]

    from models.keras_fn.attention_lenet import AttentionLeNet5
    model = AttentionLeNet5(input_shape=input_shape,
                            num_classes=num_classes,
                            attention="senet")


if __name__ == "__main__":
    main()
