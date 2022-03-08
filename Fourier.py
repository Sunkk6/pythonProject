from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def get_mse(true, pred):  # 定义损失函数（均方差）
    # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
    return np.dot((true - pred), (true - pred).T) / len(true) * 100


epoch = 3
lr = 0.01
weight_column = 10
weight = np.ones((2, weight_column))
f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = signal.square(2 * np.pi * f * t)
sin = {}
cos = {}
sin_all = 0
cos_all = 0
for i in range(weight_column):
    sin[i] = weight[0][i] * np.sin((i + 1) * 2 * np.pi / 0.8 * t)
    cos[i] = weight[1][i] * np.cos((i + 1) * 2 * np.pi / 0.8 * t)
    sin_all += sin[i]
    cos_all += cos[i]
signal_all = sin_all + cos_all

loss = get_mse(square, signal_all)

# print(grads)
plt.plot(t, square)
plt.plot(t, signal_all)
plt.ylim(-2, 2)
plt.show()
# print(len(weight))
print(loss)
# print(len(square), len(signal_all))
# for i in range(epoch):
#     weight -= lr *
#     y = y - alpha * 2 * (y - 10)
#     print("第{}次迭代：x=%f，y=%f，fxy=%f" % (i + 1, x, y, f))
