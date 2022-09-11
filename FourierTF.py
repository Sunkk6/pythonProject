import numpy as np
import tensorflow as tf


from scipy import signal
import matplotlib.pyplot as plt
import time

print(tf.__version__)

num_epoch = 100
lr = tf.constant(0.1)
weight_row = 100
weight = tf.Variable(np.ones((weight_row, 2), dtype=np.float32))
f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = tf.cast(signal.square(2 * np.pi * f * t), dtype=np.float32)


def get_mse(true, pred):
    # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
    return tf.reduce_sum(tf.square(pred - true)) / len(true)


# optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss_plt = {}
sin = {}
cos = {}
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
    grads = tape.gradient(loss, weight)
    w = tf.constant(weight)
    weight = tf.Variable(tf.subtract(weight, tf.multiply(lr, grads)))
    # optimizer.apply_gradients(grads_and_vars=zip(grads, weight))
    # time.sleep(1)
    plt.plot(t, square)
    plt.ylim(-2, 2)
    plt.plot(t, signal_all)
    plt.pause(0.002)
    print('第{}次:'.format(e + 1), 'Training loss is :', loss.numpy())
plt.figure(2)
plt.ylim(0, 0.005)
plt.plot(np.linspace(0, num_epoch, num_epoch, endpoint=False), loss_plt.values())
