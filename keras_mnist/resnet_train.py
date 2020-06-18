#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import json
import random
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras  # tf2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img

from resnet import model_depth, resnet_v2, lr_schedule
# from model import auc # tf1
from keras.metrics import AUC  # 等价于 from tf.keras.metrics import AUC
from metrics import AUC0

# Training parameters
START_EPOCH = 0
IF_FAST_RUN = False
TRAINING_EPOCHS = 50

TOTAL_TRAIN = 30000 * 0.8
TOTAL_VALIDATE = 30000 * 0.2

# constants
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),  # 整体的 accuracy
    AUC(name='auc_good_0'),  # 实际上是以 good 为 positive 的 AUC
    AUC0(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
]


def main():
    print("If in eager mode: ", tf.executing_eagerly())
    print("Use tensorflow version 2.")
    assert tf.__version__[0] == "2"

    print("Load Config ...")
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    MODEL_CKPT = CONFIG["MODEL_CKPT"]

    print("Prepare Model")
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    MODEL_CKPT = os.path.join(SAVES_DIR, MODEL_CKPT)

    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(TRAINING_EPOCHS)),
                  metrics=METRICS)
    # model.summary()
    print(MODEL_TYPE)

    print("Resume Training...")
    model_ckpt_file = MODEL_CKPT
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    # Prepare model model saving directory.
    # model_name = "%s.%03d-val_accuracy-{val_accuracy:.4f}.h5" % (MODEL_TYPE, epoch+START_EPOCH)
    model_name = "%s.start-%d-epoch-{epoch:03d}-auc_good_0-{auc_good_0:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % (
        MODEL_TYPE, START_EPOCH)
    filepath = os.path.join(SAVES_DIR, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc_good_0", verbose=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc_good_0",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [learning_rate_reduction, checkpoint]  # 不要 earlystop

    print("Training Generator...")
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='training',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Fit Model...")
    epochs = 3 if IF_FAST_RUN else TRAINING_EPOCHS
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//BATCH_SIZE,
        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
        callbacks=callbacks
    )

    print("Save Model...")
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    print("Save History...")
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
