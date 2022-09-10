from PIL import Image
from os import listdir
import numpy as np


def resize_img(input_path, out_path, x, y):
    fp = open(input_path, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    img = Image.fromarray(pic_array)
    print("修改前: ", img.size)
    new_img = img.resize((x, y))
    new_img.save(out_path)
    print("修改后: ", new_img.size)


if __name__ == '__main__':
    inpath = r"C:\Users\28972\Desktop\garbage_classification\cans"  # 在此输入图片输入路径
    outpath = r"C:\Users\28972\Desktop\garbage_classification\cans"  # 在此输入图片输出路径
    x = 448  # 图片水平长度
    y = 448  # 图片垂直长度

    for i in listdir(inpath):
        print(i)
        resize_img(inpath + '\\' + i, outpath + '\\' + i, x, y)
        print("--------------------")
    print("完成完成完成完成完成完成完成完成")

