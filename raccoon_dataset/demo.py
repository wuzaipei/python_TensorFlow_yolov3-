#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-10 14:40:33
#   Description :
#
#================================================================

import cv2
import numpy as np
from PIL import Image

file_name = "raccoon-110.jpg"
data = open("labels.txt").readlines()
for i in range(len(data)):
    image_info = data[i].split()
    if image_info[0] == file_name:
        break

image = cv2.imread(image_info[0][18:])
n_box = len(image_info[1:]) // 5 # xmin, ymin, xmax, ymax, id
for i in range(n_box):
    image = cv2.rectangle(image,(int(float(image_info[1+i*5])),
                                 int(float(image_info[2+i*5]))),
                                (int(float(image_info[3+i*5])),
                                 int(float(image_info[4+i*5]))), (255,0,0), 2)

image = Image.fromarray(np.uint8(image))
image.show()
