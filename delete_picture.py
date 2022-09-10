"""删除不合格图片"""

from pathlib import Path
import imghdr
import os

data_dir = r"C:\Users\28972\Desktop\garbage_classification"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "jpeg", "gif", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        img_size = os.path.getsize(filepath)
        # print(img_size)
        if img_size == 52 or img_size == 285 or img_size == 6033:
            os.remove(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
            os.remove(filepath)
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            os.remove(filepath)
