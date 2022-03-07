from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def get_mse(true, pred):   # 定义损失函数（均方差）
    if len(true) == len(pred):
        # return sum([(x - y) ** 2 for x, y in zip(true, pred)]) / len(true)
        return sum([(np.array(true) - np.array(pred)) ** 2]) / len(true)
    else:
        return None


f = 1 / 0.8
t = np.linspace(0, 1, 500, endpoint=False)
square = signal.square(2 * np.pi * f * t)
sin1 = np.sin(2 * np.pi / 0.8 * t - 0)
sin2 = np.sin(2 * 2 * np.pi / 0.8 * t - 0)
sin3 = np.sin(3 * 2 * np.pi / 0.8 * t - 0)
plt.plot(t, square)
plt.plot(t, sin1)
plt.ylim(-2, 2)
plt.show()
print(get_mse(square, sin1))
