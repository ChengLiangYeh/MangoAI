import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt

import torch
read_path = "/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/Masked_Dev/"
save_path ="/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/clache_DEV/"

# img = cv2.imread("/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/Masked_Dev/mask00059.jpg")
# img = cv2.resize(img)


# create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv2.imshow("new",cl1)
# cv2.waitKey(0)

# cv2.imshow('normal',img)
# print(img.shape)
# img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# #img_hsv[:,:,1] = cv2.equalizeHist(img_hsv[:,:,1])
# img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
# img_hsv_afterhistequalize_to_Schannel = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
# cv2.imshow('img_hsv_afterhistequalize_to_Schannel',img_hsv_afterhistequalize_to_Schannel)


image_path = "/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/Masked_Dev/mask00059.jpg"

jpg_list = sorted(os.listdir(read_path))
for name in jpg_list:
    now_name =read_path +name
    save_name =save_path +"clache"+ name[4:]
    print(now_name)
    bgr = cv2.imread(now_name)
    # bgr = cv2.resize(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=6.0,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(save_name,bgr)