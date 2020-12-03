#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-23-20 22:48
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import pickle
import numpy as np
import tensorflow as tf  # tensorflow >= 2.1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model


class MyLayer(Layer):
    def __init__(self, dense_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.dense_dim = dense_dim

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dense = self.add_weight("attn_vec", shape=[
                                     input_shape[1].value, self.dense_dim], initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)

    def get_config(self):
        config = {'dense_dim': self.dense_dim}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):
        # matrix multiply
        return tf.matmul(input, self.dense)


def main():
    MODEL_TYPE = "MyModel"
    epochs = 1
    ROOT_PATH = "D:\\DeepLearningData\\mnist"
    SAVES_DIR = os.path.join(ROOT_PATH, "ckpts-%s/" % MODEL_TYPE)

    METRICS = ['accuracy']

    print("Tensorflow version: ", tf.__version__)
    # Load data
    num_classes = 10
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data(
        path=os.path.join("D:\\DeepLearningData\\datasets", "mnist.npz"))
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # y_train = tf.one_hot(y_train, num_classes)
    # y_test = tf.one_hot(y_test, num_classes)

    ckpt_name = "%s-epoch-{epoch:03d}-acc-{acc:.4f}.h5" % MODEL_TYPE
    filepath = os.path.join(SAVES_DIR, ckpt_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="acc", verbose=1)
    csv_logger = CSVLogger("./log/training.log.csv", append=True)
    earlystop = EarlyStopping(patience=10)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, csv_logger, lr_reducer]

    with tf.device('/device:GPU:0'):
        # build a 2-layer FC model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss=SparseCategoricalCrossentropy(),
                      metrics=METRICS)

        history = model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=epochs
            # callbacks=callbacks
        )

    with open("./log/history.pickle", 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.evaluate(X_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()
