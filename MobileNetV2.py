import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_epoch = 10
batch_size = 32
learning_rate = 0.001

dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(
    batch_size)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
model = tf.keras.applications.MobileNetV2(weights=None, classes=10)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(dataset, epochs=num_epoch, callbacks=[tensorboard_callback])
# print(model.evaluate())

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

# for image, label in dataset.take(2):
#     for i in range(4):
#         plt.imshow(image.numpy()[i, :, :, :])
#         plt.colorbar()
#         plt.show()

# print(dataset)
