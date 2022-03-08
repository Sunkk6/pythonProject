import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import time

num_epoch = 3
lr = 0.01
weight_row = 10
weight = tf.Variable(np.ones((weight_row, 2), dtype=np.float32))

f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = tf.cast(signal.square(2 * np.pi * f * t), dtype=np.float32)


def get_mse(true, pred):  # 定义损失函数（均方差）
    # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
    return tf.reduce_sum(tf.square(pred - true)) / len(true) * 100


optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

sin = {}
cos = {}
sin_all = 0
cos_all = 0
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        for i in range(weight_row):
            sin[i] = weight[i][0] * np.sin((i + 1) * 2 * np.pi / 0.8 * t)
            cos[i] = weight[i][1] * np.cos((i + 1) * 2 * np.pi / 0.8 * t)
            sin_all += sin[i]
            cos_all += cos[i]
        signal_all = sin_all + cos_all
        loss = get_mse(square, sin[0])
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, weight)
    # TensorFlow自动根据梯度更新参数
    # optimizer.apply_gradients(grads_and_vars=zip(grads, weight))

plt.plot(t, square)
plt.plot(t, signal_all)
plt.ylim(-2, 2)
plt.show()
print(weight)
