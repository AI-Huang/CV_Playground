#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-04-20 20:57
# @Author  : Kelly Hwong (you@example.org)

"""MNIST training code

# Environments

tensorflow>=2.1.0

"""

import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from data.tf_fn.load_data import load_data
from model_utils.keras_fn import create_model, create_optimizer
from model_utils.keras_fn.lr_schedules import polynomial_schedule

# Training settings
batch_size = 32
if_fast_run = False
training_epochs = 100


def time_string():
    # datetime.now().strftime('%Y-%m-%d %H:%M')
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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
    # parser.add_argument('--model', default='tacotron')
    # parser.add_argument(
    #     '--name', help='Name of the run. Used for logging. Defaults to model name.')

    parser.add_argument('--model_name', type=str, dest='model_name',
                        action='store', default="LeNet5", help="""model_name, one of ["LeNet5", "AttentionLeNet5"].""")

    # Attention parameters
    parser.add_argument('--attention', type=string2bool, dest='attention',
                        action='store', default=False, help='attention, if true, the model will use Attention.')

    parser.add_argument('--attention_type', type=str, dest='attention_type',
                        action='store', default="official", help="""attention_type, one of ["official", "senet"], used only when --attention flag is set.""")

    args = parser.parse_args()

    if args.attention_type not in ["official", "senet"]:
        raise ValueError(
            f"""args.attention_type {args.attention_type} NOT in ["official", "senet"]""")

    return args


def main():
    args = cmd_parser()
    model_name = args.model_name

    # Check inputs
    resnetxx = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    available_models = ["LeNet5", "AttentionLeNet5", "LeCunLeNet5"] + resnetxx
    if args.model_name not in available_models:
        raise ValueError(
            f"""args.model_name {args.model_name} NOT in {available_models}""")

    if args.attention:
        if args.attention_type == "senet":
            model_name = "AttentionLeNet5_SeNet"
        elif args.attention_type == "official":
            model_name = "AttentionLeNet5_Official"

    # Config paths
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", "mnist")

    # Prepare data
    dataset = "mnist"
    (train_images, train_labels), (test_images, test_labels) = load_data(
        dataset=dataset, if_categorical=True, if_expand_dims=True, if_normalized=False)

    input_shape = train_images.shape[1:]
    num_classes = train_labels.shape[1]

    # Setup model
    if model_name not in resnetxx:
        model = create_model(
            model_name, input_shape=input_shape, num_classes=num_classes)
        optimizer = create_optimizer("Adam", learning_rate=0.001)

    # Preprocessing and choose optimizer for ResNet18
    elif model_name in resnetxx:
        model_core = create_model(
            model_name, input_shape=(32, 32, 1), num_classes=num_classes)

        input_ = tf.keras.layers.Input(input_shape, dtype=tf.uint8)
        x = tf.cast(input_, tf.float32)
        # padding 28x28 to 32x32
        x = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])
        x = model_core(x)
        model = tf.keras.Model(inputs=[input_], outputs=[x])
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=polynomial_schedule(0))

    subfix = os.path.join(model_name, date_time)
    ckpt_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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

    lr_scheduler = LearningRateScheduler(polynomial_schedule, verbose=1)
    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")
    # without .h5 extension
    ckpt_filename = "%s-epoch-{epoch:03d}-categorical_accuracy-{categorical_accuracy:.4f}" % model_name
    ckpt_filepath = os.path.join(ckpt_dir, ckpt_filename)
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_filepath,
        monitor="categorical_accuracy",
        verbose=1,
        save_weights_only=True
    )

    callbacks = [csv_logger, lr_scheduler,
                 checkpoint_callback, tensorboard_callback]

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
