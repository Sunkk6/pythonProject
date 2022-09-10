import datetime
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image

model = tf.saved_model.load("saved/mnist")

pred_img = cv2.imread("Picture_predict/2.jpg", 0)

pred_img = cv2.resize(pred_img, (28, 28))
reshape_pred_img = pred_img.reshape(-1, 28, 28, 1)
input_1 = tf.convert_to_tensor(np.array(reshape_pred_img), dtype=np.float32)
# cv2.imshow('ppp', input_1)
print(input_1)
logits = model(input_1)
# print(pred_img_tensor)
# result = [np.argmax(i) for i in logits]

# img_path = "Picture_predict/5.jpg"
# img = image.load_img(img_path, target_size=(28, 28))
# pred_img = image.img_to_array(img)
# input_1 = np.expand_dims(pred_img, axis=0)
# # pred_img = tf.keras.applications.mobilenet_v2.preprocess_input(pred_img)
# preds = model(input_1)
# plt.imshow(input_1)
# plt.show()

result = np.argmax(np.array(logits))
print(logits, result)

