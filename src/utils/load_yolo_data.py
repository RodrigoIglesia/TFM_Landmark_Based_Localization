import os
import sys
import cv2
from ultralytics.utils.plotting import Annotator

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.insert(0, src_dir)

img = cv2.imread('dataset/waymo_yolo_format/images/train/00000_00000_camera1.jpg')


# Showing image shape
print('Image shape:', img.shape)  # tuple of (800, 1360, 3)

# Getting spatial dimension of input image
dh, dw, _ = img.shape

# Showing height an width of image
print('Image height={0} and width={1}'.format(dh, dw))  # 800 1360

fl = open('dataset/waymo_yolo_format/labels/train/00000_00000_camera1.txt', 'r')
data = fl.readlines()
fl.close()

for dt in data:
    print(dt)
    # Split string to float
    _, x, y, w, h = map(float, dt.split(' '))

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)

cv2.imshow("image", img)
cv2.waitKey(0)

