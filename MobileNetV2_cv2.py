import cv2
import keras.models
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

model = keras.models.load_model("saved_server/MobileNetV2.h5")

capture = cv2.VideoCapture(0)  # 创建一个VideoCapture对象
k = 0
while True:
    ret, frame = capture.read()  # 一帧一帧读取视频
    # cv2.imshow('frame', frame)  # 显示结果
    # print(frame.shape)
    k = k + 1
    if k == 50:
        pred_img = cv2.resize(frame, (224, 224))
        cv2.imshow('pred_img', pred_img)  # 显示结果
        pred_img = np.float32(pred_img / 255.0)
        reshape_pred_img = pred_img.reshape(-1, 224, 224, 3)
        input_1 = tf.convert_to_tensor(np.array(reshape_pred_img), dtype=np.float32)
        logits = model(input_1)
        classes_num = tf.argmax(logits, axis=1)
        logits_array = np.array(logits)
        print("置信度:", logits_array[0][int(np.array(classes_num))])
        if classes_num == 0:
            print('电池')
        elif classes_num == 1:
            print('有机物')
        elif classes_num == 2:
            print('棕色玻璃')
        elif classes_num == 3:
            print('纸板')
        elif classes_num == 4:
            print('衣服')
        elif classes_num == 5:
            print('绿色玻璃')
        elif classes_num == 6:
            print('金属')
        elif classes_num == 7:
            print('纸张类')
        elif classes_num == 8:
            print('塑料')
        elif classes_num == 9:
            print('鞋子')
        elif classes_num == 10:
            print('一般垃圾', )
        elif classes_num == 11:
            print('白色玻璃')
        k = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q停止
        break
capture.release()  # 释放cap,销毁窗口
cv2.destroyAllWindows()
