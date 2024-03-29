import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

# === dataset ===
with np.load(r'C:\Users\28972\.keras\datasets\mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)
print(x_test.shape)

# === model: CNN ===
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.summary()

# === train ===
model.fit(x=x_train, y=y_train,
          batch_size=512,
          epochs=10,
          validation_data=(x_test, y_test))

# === pred ===
y_pred = model.predict(x_test)
print(y_pred.argmax(axis=1))

# === 混淆矩阵：真实值与预测值的对比 ===
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
con_mat = confusion_matrix(y_test, y_pred.argmax(axis=1))

kind = ['zero', 'one', 'two', 'three', 'four', 'four', 'four', 'four', 'four', 'four']

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
conf_df = pd.DataFrame(con_mat_norm, index=kind, columns=kind)  # 将矩阵转化为 DataFrame

# === plot ===
plt.figure(figsize=(8, 8))
sns.heatmap(conf_df, annot=True, cmap='Blues')

plt.ylim(10, 0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
