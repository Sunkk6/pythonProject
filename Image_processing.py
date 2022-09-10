import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def processing_data(data_path):
    train_data = ImageDataGenerator(
        rescale=1. / 225,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1, )
    validation_data = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1)
    train_generator = train_data.flow_from_directory(
        data_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='training', seed=0)
    validation_generator = validation_data.flow_from_directory(
        data_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='validation', seed=0)

    return train_generator, validation_generator
