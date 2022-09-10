import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import time

num_epoch = 100
weight_row = 100
learning_rate = 0.1
lr = tf.constant(learning_rate, dtype=np.float32)
theta = tf.constant(np.ones((weight_row, 2), dtype=np.float32))
weight = tf.Variable(np.ones((weight_row, 2), dtype=np.float32))
f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = tf.cast(signal.square(2 * np.pi * f * t), dtype=np.float32)


def get_mse(true, pred):
    # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
    return tf.reduce_sum(tf.square(pred - true)) / len(true)


# optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
sin = {}
cos = {}
grad = {}
loss_plt = {}
for e in range(num_epoch):
    plt.clf()
    # signal_all = tf.constant(np.zeros(500))
    sin_all = 0
    cos_all = 0
    with tf.GradientTape() as tape:
        for i in range(weight_row):
            sin[i] = weight[i][0] * np.sin((i + 1) * 2 * np.pi / 0.8 * t)
            cos[i] = weight[i][1] * np.cos((i + 1) * 2 * np.pi / 0.8 * t)
            sin_all += sin[i]
            cos_all += cos[i]
        signal_all = sin_all + cos_all
        loss = get_mse(square, signal_all)
    loss_plt[e] = np.array(loss)
    grads = tape.gradient(loss, weight)  # 求梯度
    grad[e] = np.array(grads)
    grad_all = 0
    for k in range(e+1):
        grad_all += tf.reduce_sum(tf.square(grad[k]))
    theta = tf.constant(tf.sqrt(tf.divide(grad_all, e+1)))
    theta = tf.multiply(theta, learning_rate / 2)
    weight = tf.Variable(tf.subtract(weight, tf.multiply(tf.divide(lr, theta), grads)))
    # optimizer.apply_gradients(grads_and_vars=zip(grads, weight))
    plt.plot(t, square)
    plt.ylim(-2, 2)
    plt.plot(t, signal_all)
    plt.pause(0.002)
    print('第{}次:'.format(e+1), 'Training loss is :', loss.numpy(), 'theta :', theta.numpy())
    # print(tf.divide(lr, theta))
plt.figure(2)
plt.ylim(0, 0.005)
plt.plot(np.linspace(0, num_epoch, num_epoch, endpoint=False), loss_plt.values())
# print(np.array(loss_plt.values()))



