import datetime
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# pip install -i https://pypi.doubanio.com/simple/ 包名

"""================可视化==================="""
# tensorboard --logdir logs/fit
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m月%d日-%H时%M分%S秒")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# model.fit(train_dataset, epochs=3, callbacks=[tensorboard_callback])
"""========================================"""

"""禁用GPU"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# tf.debugging.set_log_device_placement(True)

"""用老版本"""
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# data = tf.constant([1, 2])
# print(data.numpy())

# arr_list = np.arange(0, 100).astype(np.float64)
# shape = arr_list.shape
# dataset = tf.data.Dataset.from_tensor_slices(arr_list)
# dataset_interator = dataset.shuffle(shape[0]).batch(10)
#
# def model(xs):
#     outputs = tf.multiply(xs, 0.1)
#     return outputs
#
# # print(dataset)
#
# for it in dataset_interator:
#     logits = model(it)
#     print(logits)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(256, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(2, activation="softmax"))
# data = load_iris()
# iris_target = data.target
# iris_data = data.data
# print(iris_target)
# print(iris_target.shape)

# print(tf.__version__)

# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# # print(tf.reshape(x, [3,2]))
# print(tf.transpose(x))

# print(datetime.datetime.now().strftime("%Y年%m月%d日-%H时%M分%S秒"))
# print(datetime.datetime.now())

label_100 = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train',
             28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle',
             71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose',
             87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit',
             21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster',
             49: 'mountain',
             56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee',
             20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl',
             83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel',
             18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket',
             95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox',
             32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp',
             26: 'crab',
             48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge',
             2: 'baby',
             41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain',
             91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}

# cifar100 = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(y_train[201])

# model = tf.keras.applications.resnet.ResNet152()
# print(model.summary())

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # print(x_train.shape)
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = tf.expand_dims(x_train, -1)
# # y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=10))
# x_test = tf.expand_dims(x_test, -1)
# y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=10))
# bacth_size = 128
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# # for image, label in train_dataset:
#     plt.title(image.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()
# for k in range(19, 22):
#     plt.title(label_100[int(y_train[k])])
#     plt.imshow(x_train[k][:, :, :])
#     plt.colorbar()
#     plt.show()
#     print(label_100[k])
# print(int(y_train[4]))
