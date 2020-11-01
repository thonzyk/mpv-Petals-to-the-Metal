# -*- coding: utf-8 -*-
"""Set of different models returned by each function"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
# from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from .constants import IMAGE_SIZE, CLASSES

# INITIALIZATION
# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

# Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on
BATCH_SIZE = 16 * strategy.num_replicas_in_sync


# MODELS
def tutorial_model():
    """Copy of the model from the tutorial notebook
    https://www.kaggle.com/ryanholbrook/create-your-first-submission
    """
    with strategy.scope():
        pretrained_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=[*IMAGE_SIZE, 3]
        )
        pretrained_model.trainable = False

        model = tf.keras.Sequential([
            # To a base pretrained on ImageNet to extract features from images...
            pretrained_model,
            # ... attach a new head to act as a classifier.
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])

    return model


def pretrainded_model(type: str, trainable=False):
    with strategy.scope():
        if type == 'VGG16':
            pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'VGG19':
            pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet121':
            pretrained_model = DenseNet121(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet169':
            pretrained_model = DenseNet169(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet201':
            pretrained_model = DenseNet201(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

        pretrained_model.trainable = trainable

        model = Sequential([
            # To a base pretrained on ImageNet to extract features from images...
            pretrained_model,
            # ... attach a new head to act as a classifier.
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax', use_bias=False)
        ])

    return model


def vgg16(trainable=False):
    return pretrainded_model('VGG16', trainable)


def vgg19(trainable=False):
    return pretrainded_model('VGG19', trainable)


def densenet121(trainable=False):
    return pretrainded_model('DenseNet121', trainable)


def densenet169(trainable=False):
    return pretrainded_model('DenseNet169', trainable)


def densenet201(trainable=False):
    return pretrainded_model('DenseNet201', trainable)
