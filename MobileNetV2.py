import datetime
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


num_epoch = 10
batch_size = 32
learning_rate = 0.001
directory = r"C:\Users\28972\Desktop\training-set"

dataset = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=2,
    validation_split=0.1,
    subset="training",
    interpolation='bilinear',
    # crop_to_aspect_ratio=True,
)

# dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label))

# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# dataset = tf.keras.applications.mobilenet_v2.preprocess_input(dataset)

# for images, labels in dataset.take(1):
#     for i in range(32):
#         plt.imshow(images.numpy()[i, :, :, :])
#         plt.show()
#     print(labels.numpy())

model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, classes=12)
print(model.summary())
# # input_xs = tf.keras.Input([224, 224, 3])
# # conv = tf.keras.layers.Conv2D(32, 3, padding="SAME", activation=tf.nn.relu)(input_xs)
# # conv = tf.keras.layers.BatchNormalization()(conv)
# # conv = tf.keras.layers.Conv2D(64, 3, padding="SAME", activation=tf.nn.relu)(conv)
# # conv = tf.keras.layers.MaxPool2D(strides=[1, 1])(conv)
# # conv = tf.keras.layers.Conv2D(128, 3, padding="SAME", activation=tf.nn.relu)(conv)
# # flat = tf.keras.layers.Flatten()(conv)
# # dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
# # logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense)
# # model = tf.keras.Model(inputs=input_xs, outputs=logits)


# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# model.compile(optimizer=optimizer, loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
#
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# model.fit(dataset, epochs=num_epoch, callbacks=[tensorboard_callback])
# # print(model.evaluate())
# tf.saved_model.save(model, "saved/2")
# model.save('saved/MobileNetV2_imagenet.h5')
# for e in range(num_epoch):
#     for images, labels in dataset:
#         with tf.GradientTape() as tape:
#             labels_pred = model(images, training=True)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
#             loss = tf.reduce_mean(loss)
#             print("loss %f" % loss.numpy())
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
#     print(labels_pred)

# for image, label in dataset.take(1):
#     for i in range(32):
#         plt.imshow(image.numpy()[i, :, :, :])
#         plt.show()
# plt.colorbar()
# print(model.summary())
# print(dataset)
