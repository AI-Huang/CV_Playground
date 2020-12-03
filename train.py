#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-04-20 20:57
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org
"""MNIST training code

## Environments

tensorflow>=2.1.0

"""
import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils.dir_utils import makedir_exist_ok

# Training settings
batch_size = 32
if_fast_run = False
training_epochs = 100


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    def string2bool(string):
        """string2bool
        """
        if string not in ["False", "True"]:
            raise argparse.ArgumentTypeError(
                f"""input(={string}) NOT in ["False", "True"]!""")
        if string == "False":
            return False
        elif string == "True":
            return True

    parser.add_argument('--model_type', type=str, dest='model_type',
                        action='store', default="LeNet5", help="""model_type, one of ["LeNet5", "AttentionLeNet5"].""")

    # Attention parameters
    parser.add_argument('--attention', type=string2bool, dest='attention',
                        action='store', default=False, help='attention, if true, the model will use Attention.')

    parser.add_argument('--attention_type', type=str, dest='attention_type',
                        action='store', default="official", help="""attention_type, one of ["official", "senet"], used only when --attention flag is set.""")

    args = parser.parse_args()

    # Check inputs
    if args.model_type not in ["LeNet5", "AttentionLeNet5"]:
        raise ValueError(
            f"""args.model_type {args.model_type} NOT in ["LeNet5", "AttentionLeNet5"]""")
    if args.attention_type not in ["official", "senet"]:
        raise ValueError(
            f"""args.attention_type {args.attention_type} NOT in ["official", "senet"]""")

    return args


def main():
    args = cmd_parser()
    model_type = args.model_type

    if args.attention:
        if args.attention_type == "senet":
            model_type = "AttentionLeNet5-SeNet"
        elif args.attention_type == "official":
            model_type = "AttentionLeNet5-Official"

    # Config paths
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", "mnist")

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

    # Create model
    if model_type == "ResNet20v2":  # "ResNet20v2",  "ResNet56v2"
        from keras_fn.resnet import model_depth, resnet_v2, lr_schedule
        depth = model_depth(n=2, version=2)
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))
        save_dir = os.path.join(".", 'saved_models')
        model_save_name = "keras_cifar10_trained_model.h5"
    elif model_type == "LeNet5":
        from keras_fn.lenet import LeNet5
        model = LeNet5(input_shape=input_shape, num_classes=num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif model_type.startswith("AttentionLeNet5"):
        from keras_fn.attention_lenet import AttentionLeNet5
        model = AttentionLeNet5(input_shape=input_shape,
                                num_classes=num_classes,
                                attention="senet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    subfix = os.path.join(model_type, date_time)
    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs", subfix))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    padded = False
    if model_type == "ResNet20v2":
        padded = True  # True
    if padded:
        train_images = np.load("./mnist_train_images_padded.npy")

    loss = tf.keras.losses.CategoricalCrossentropy(
        name="categorical_crossentropy")
    from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
    metrics = [BinaryAccuracy(name="binary_accuracy"),
               CategoricalAccuracy(name="categorical_accuracy")]

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # Define callbacks
    from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, TensorBoard, ModelCheckpoint

    # lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")
    # without .h5 extension
    ckpt_filename = "%s-epoch-{epoch:03d}-binary_accuracy-{binary_accuracy:.4f}" % model_type
    ckpt_filepath = os.path.join(ckpt_dir, ckpt_filename)
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_filepath,
        monitor="binary_accuracy",
        verbose=1,
        save_weights_only=True
    )

    callbacks = [csv_logger, tensorboard_callback, checkpoint_callback]

    # Fit model
    epochs = 3 if if_fast_run else training_epochs
    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
