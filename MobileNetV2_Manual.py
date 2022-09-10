from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras_preprocessing.image import ImageDataGenerator

# from keras.optimizers import Adam
import keras
from keras.utils.vis_utils import plot_model
from keras import backend as back
import datetime
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
# import cv2
from tensorflow.keras.preprocessing import image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_epoch = 200
batch_size = 32
learning_rate = 0.001

classes = 12
dropout_rate = 0

directory = r"C:\Users\28972\Desktop\garbage_classification"
# directory = r"C:\Users\28972\Desktop\垃圾数据集"

# directory = r"/tmp/pycharm_project_7/garbage_classification/garbage_classification"

train_data = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # horizontal_flip=True,
    # vertical_flip=True,
    validation_split=0.2)
validation_data = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2)
train_generator = train_data.flow_from_directory(
    directory, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='training', seed=0)
validation_generator = validation_data.flow_from_directory(
    directory, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='validation', seed=0)


# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     labels='inferred',
#     label_mode='categorical',
#     class_names=None,
#     color_mode='rgb',
#     batch_size=batch_size,
#     image_size=(224, 224),
#     shuffle=True,
#     seed=2,
#     validation_split=0.1,
#     subset="training",
#     interpolation='bilinear',
#     # crop_to_aspect_ratio=True,
# )

# dataset = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True)
# dataset1 = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label))


# for images, labels in train_generator:
#     for i in range(16):
#         plt.imshow(images[i, :, :, :])
#         plt.show()
#     print(images)


#
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    '''确保四舍五入的降幅不超过10%'''
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6"""
    return back.relu(x, max_value=6.0)


def conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if back.image_data_format() == 'channels_first' else -1
    # print(channel_axis)

    x = Conv2D(filters, kernel, padding='same', strides=strides, use_bias=True)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if back.image_data_format() == 'channels_first' else -1
    # 深度
    tchannel = back.int_shape(inputs)[channel_axis] * t
    # 宽度
    cchannel = int(filters * alpha)

    x = conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same', use_bias=True)(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    x = bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def MobileNetv2(input_shape, k, alpha=1.0):
    inputs = Input(shape=input_shape)

    first_filters = make_divisible(32 * alpha, 8)
    # print(first_filters)
    x = conv_block(inputs, first_filters, (3, 3), strides=(2, 2))
    # x = Conv2D(first_filters, (3, 3), padding='same', strides=(2, 2))(inputs)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation(relu6)(x)

    x = inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = conv_block(x, last_filters, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_filters))(x)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same', use_bias=True)(x)

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='saved/MobileNetv2.png', show_shapes=True)

    return model


# def get_config(self):
#     config = super().get_config().copy()
#     config.update({
#         'inputs': self.inputs,
#         'filters': self.filters,
#         'kernel': self.kernel,
#         'strides': self.strides,
#         'input_shape': self.input_shape,
#         'dropout': self.dropout,
#         'alpha': self.alpha,
#         'k': self.k,
#         't': self.t,
#         'n': self.n,
#         's': self.s,
#         'r': self.r,
#     })
#     return config
#
#
# # @classmethod
# def from_config(cls, config):
#     return cls(**config)


model = MobileNetv2((224, 224, 3), classes, 1)
# print(model.summary())

optimizer = keras.optimizers.adam_v2.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

'''accuracy'''
'''sparse_accuracy'''

early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0,
                                               patience=3, verbose=1, mode='max',
                                               baseline=None, restore_best_weights=False)

# log_dir = "/tmp/pycharm_project_7/logs/fit_server/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
log_dir = "logs/fit2/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_generator, epochs=num_epoch, verbose=1, validation_data=validation_generator,
                    callbacks=[tensorboard_callback, early_stopping])
model.evaluate(validation_generator)
# history = model.fit(dataset1, epochs=num_epoch)

# model.save('saved/MobileNetV2_4_8.h5')
# model.save('/tmp/pycharm_project_7/saved_server/MobileNetV2_4_16.h5')
# tf.saved_model.save(model, "saved/oo1")
# tf.saved_model.save(model, "/tmp/pycharm_project_7/saved_server/1")
