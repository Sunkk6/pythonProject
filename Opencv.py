import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 调用cv.calcHist方法计算，和第一个函数一样的结果
def Gray_hist(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    return np.array(hist)


# 读取图像
src = cv.imread(r"C:\Users\28972\Desktop\WIN_20220330_16_02_59_Pro.jpg")
gray_img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("gray_img", gray_img)

# 画出原图的灰度图的直方图
hist1 = Gray_hist(gray_img)
hist1 = hist1.flatten()
plt.subplot(211)
plt.bar(range(256), hist1)
plt.title("gray_img hist")

equalizeHist_img = cv.equalizeHist(gray_img)
cv.imshow("equalizeHist_img", equalizeHist_img)
cv.imwrite("test2.jpg", equalizeHist_img)
hist2 = Gray_hist(equalizeHist_img)
hist2 = hist2.flatten()
plt.subplot(212)
plt.bar(range(256), hist2)
plt.title("equalizeHist_img hist")

duibi1 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
print("HISTCMP_CORREL", duibi1)
duibi1 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
print("HISTCMP_CHISQR", duibi1)
duibi1 = cv.compareHist(hist1, hist2, cv.HISTCMP_INTERSECT)
print("HISTCMP_INTERSECT", duibi1)
duibi1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
print("HISTCMP_BHATTACHARYYA", duibi1)

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
