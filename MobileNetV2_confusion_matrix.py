import datetime

import keras.models
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import cv2
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# model = tf.saved_model.load("/tmp/pycharm_project_7/saved_server/1")
# model = tf.saved_model.load("saved/2")
# model = tf.saved_model.load("saved/oo1")
# model = keras.models.load_model("saved_server/MobileNetV2.h5")
model = keras.models.load_model("saved/MobileNetV2_5_16_81e.h5")

batch_size = 1
directory1 = r"C:\Users\28972\Desktop\garbage_classification - 副本 (2)"
# directory1 = r"C:\Users\28972\Desktop\garbage_classification_test"
# dataset1 = tf.keras.preprocessing.image_dataset_from_directory(
#     directory1,
#     labels='inferred',
#     label_mode='categorical',
#     class_names=None,
#     color_mode='rgb',
#     batch_size=batch_size,
#     image_size=(224, 224),
#     shuffle=False,
#     seed=2,
#     validation_split=0,
#     # subset="training",
#     interpolation='bilinear',
#     # crop_to_aspect_ratio=True,
# )
#
# dataset11 = dataset1.map(
#     lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label))

# train_data = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.1,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=True,
#     validation_split=0.1)
validation_data = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1)
# train_generator = train_data.flow_from_directory(
#     directory1, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='training', seed=0)
validation_generator = validation_data.flow_from_directory(
    directory1, target_size=(224, 224), batch_size=batch_size, shuffle=False, class_mode='categorical',
    subset='validation', seed=0)

# numm = 1
# for images, labels in dataset11.take(numm):
#     numpy_labels = labels.numpy()

# num = 0
# label_all = []
# images, labels = tuple(zip(*validation_generator))
# for images, labels in validation_generator:
#     num = num + 1
#     print(num)
#     print(labels.argmax(axis=1))
#     label_all += labels.argmax(axis=1)
# images, labels = validation_generator
# print(validation_generator)
# AA = np.array(labels).reshape(2086, 12)
# BB = model.predict(dataset11)
# print(np.array(labels).reshape(360, 12).argmax(axis=1), model.predict(dataset11).argmax(axis=1))
con_mat = confusion_matrix(np.array(validation_generator.labels),
                           model.predict(validation_generator).argmax(axis=1))

kind = ['battery', 'biological', 'cans', 'cardboard', 'clothes', 'glass-bottle', 'mask', 'mobile-phone',
        'plastic-bottle', 'shoes', 'toilet-paper', 'toothbrush']

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=3)
conf_df = pd.DataFrame(con_mat_norm, index=kind, columns=kind)  # 将矩阵转化为 DataFrame
# === plot ===
plt.figure(figsize=(8, 8))
sns.heatmap(conf_df, annot=True, cmap='Blues')
plt.ylim(12, 0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# print(model.predict(dataset11.take(1)))
