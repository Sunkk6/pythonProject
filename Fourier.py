from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def get_mse(true, pred):  # 定义损失函数（均方差）
    if len(true) == len(pred):
        # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
        return np.sum([(np.array(true) - np.array(pred)) ** 2]) / len(true) * 100
    else:
        return None


epoch = 3
weight = np.ones(10)
sin = {}
f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = signal.square(2 * np.pi * f * t)
sin_all = 0
for i in range(len(weight)-1):
    sin[i] = weight[i] * np.sin((i+1) * 2 * np.pi / 0.8 * t)
    sin_all = sin_all + sin[i]
# for i in range(len(weight)):

loss = get_mse(square, sin_all)
plt.plot(t, square)
plt.plot(t, sin_all)
plt.ylim(-2, 2)
plt.show()
# print(weight)
print(loss)
# for i in range(epoch):
#     x = x - alpha * 2 * (x - 10)
#     y = y - alpha * 2 * (y - 10)
#     f = fxy(x, y)
#     print("第{}次迭代：x=%f，y=%f，fxy=%f" % (i + 1, x, y, f))
