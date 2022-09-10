import datetime

import keras.models
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image

# model = tf.saved_model.load("/tmp/pycharm_project_7/saved_server/1")
# model = tf.saved_model.load("saved/2")
# model = tf.saved_model.load("saved/oo1")
model = keras.models.load_model("saved_server/MobileNetV2_4_16.h5")
# model = keras.models.load_model("saved/MobileNetV2_1.h5")

# pred_img = cv2.imread("Picture_predict/battery945.jpg", 1)
# pred_img = cv2.imread("Picture_predict/white-glass775.jpg", 1)
# pred_img = cv2.imread("Picture_predict/biological985.jpg", 1)
# pred_img = cv2.imread("Picture_predict/brown-glass607.jpg", 1)
# pred_img = cv2.imread("Picture_predict/cardboard891.jpg", 1)
# pred_img = cv2.imread("Picture_predict/clothes5325.jpg", 1)
# pred_img = cv2.imread("Picture_predict/green-glass629.jpg", 1)
# pred_img = cv2.imread("Picture_predict/metal769.jpg", 1)
# pred_img = cv2.imread("Picture_predict/plastic865.jpg", 1)
# pred_img = cv2.imread("Picture_predict/shoes1977.jpg", 1)
# pred_img = cv2.imread("Picture_predict/dandelion1.jpeg", 1)
# pred_img = cv2.imread("Picture_predict/paper854.jpg", 1)
# pred_img = cv2.imread("Picture_predict/plastic413.jpg", 1)
#
# pred_img = cv2.resize(pred_img, (224, 224))
# pred_img = np.float32(pred_img / 255.0)
# # pred_img = pred_img.map(lambda img: tf.image.resize(img, (224, 224)) / 255.0)
# reshape_pred_img = pred_img.reshape(-1, 224, 224, 3)
# input_1 = tf.convert_to_tensor(np.array(reshape_pred_img), dtype=np.float32)
# # input_1 = tf.keras.applications.mobilenet_v2.preprocess_input(input_1)
# # plt.imshow(input_1)
# # plt.show()
# # cv2.imshow('1', pred_img)
# # print(input_1[0][100])
# logits = model(input_1)
# classes_num = tf.argmax(logits, axis=1)
# # print(classes_num)
# if classes_num == 0:
#     print('电池')
# elif classes_num == 1:
#     print('有机物')
# elif classes_num == 2:
#     print('棕色玻璃')
# elif classes_num == 3:
#     print('纸板')
# elif classes_num == 4:
#     print('衣服')
# elif classes_num == 5:
#     print('绿色玻璃')
# elif classes_num == 6:
#     print('金属')
# elif classes_num == 7:
#     print('纸张类')
# elif classes_num == 8:
#     print('塑料')
# elif classes_num == 9:
#     print('鞋子')
# elif classes_num == 10:
#     print('一般垃圾', )
# elif classes_num == 11:
#     print('白色玻璃')

# img_path = "Picture_predict/battery945.jpg"
# img_path = "Picture_predict/white-glass775.jpg"
# img_path = "Picture_predict/biological1.jpg"
# img_path = "Picture_predict/biological985.jpg"
# img_path = "Picture_predict/brown-glass607.jpg"
# img_path = "Picture_predict/cardboard891.jpg"
# img_path = "Picture_predict/clothes5325.jpg"
# img_path = "Picture_predict/plastic865.jpg"
# img_path = "Picture_predict/roses2.jpg"
#
# img_path = r"C:\Users\28972\Desktop\garbage_classification_test\clothes\clothes5004.jpg"
# # img_path = "/tmp/pycharm_project_7/Picture_predict/cardboard344.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# pred_img = image.img_to_array(img)
# pred_img = np.float32(pred_img / 255.0)
# pred_img = np.expand_dims(pred_img, axis=0)
# # pred_img = tf.keras.applications.mobilenet_v2.preprocess_input(pred_img)
# preds = model(pred_img)
# # plt.imshow(pred_img)
# # plt.show()
# # cv2.imshow('1', img)
# # results = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)
#
# print(preds)
# # print(tf.argmax(preds, axis=1))

batch_size = 32
directory1 = r"C:\Users\28972\Desktop\garbage_classification"
directory2 = r"C:\Users\28972\Desktop\garbage_classification_test"
dataset1 = tf.keras.preprocessing.image_dataset_from_directory(
    directory1,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=2,
    validation_split=0,
    # subset="training",
    interpolation='bilinear',
    # crop_to_aspect_ratio=True,
)
dataset2 = tf.keras.preprocessing.image_dataset_from_directory(
    directory2,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=2,
    validation_split=0,
    # subset="training",
    interpolation='bilinear',
    # crop_to_aspect_ratio=True,
)

dataset11 = dataset1.map(
    lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label))

dataset22 = dataset2.map(
    lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label))

print(model.evaluate(dataset11), model.evaluate(dataset22))
# print(model.evaluate(dataset11))
