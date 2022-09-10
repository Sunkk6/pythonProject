import datetime
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.expand_dims(x_train, -1)
y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=10))
x_test = tf.expand_dims(x_test, -1)
y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=10))
bacth_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(bacth_size).shuffle(bacth_size * 10)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(bacth_size)


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, filter, kernel_size):
        super().__init__()
        self.filter = filter
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.weight = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_shape[-1], self.filter]))
        self.bias = tf.Variable(tf.random.normal([self.filter]))
        super().build(input_shape)

    def call(self, input_tensor):
        conv = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, self.bias)
        out = tf.nn.relu(conv) + conv
        return out


input_xs = tf.keras.Input([28, 28, 1])
conv = tf.keras.layers.Conv2D(32, 3, padding="SAME", activation=tf.nn.relu)(input_xs)
conv = MyLayer(32, 3)(conv)
conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Conv2D(64, 3, padding="SAME", activation=tf.nn.relu)(conv)
conv = tf.keras.layers.MaxPool2D(strides=[1, 1])(conv)
conv = tf.keras.layers.Conv2D(128, 3, padding="SAME", activation=tf.nn.relu)(conv)
flat = tf.keras.layers.Flatten()(conv)
dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense)
model = tf.keras.Model(inputs=input_xs, outputs=logits)
# print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])

model.evaluate(train_dataset)

tf.saved_model.save(model, "saved/mnist")
# score = model.evaluate(test_dataset)
# print("last score:", score)
