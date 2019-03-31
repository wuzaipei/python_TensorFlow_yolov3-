#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.

#   Editor      : VIM
#   File name   : fuck.py
#   Author      : YunYang1994
#   Created date: 2019-01-23 10:21:50
#   Description :

#================================================================
import time

from matplotlib.pyplot import plot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names('./data/coco_wzp.names')
num_classes = len(classes)
image_path = "./raccoon_dataset/images/raccoon-33.jpg"  # 181,

model_path = "./model/yolov3_gpu_nms.pb"

img = Image.open(image_path)
img_resized = np.array(img.resize(size=(IMAGE_W, IMAGE_H)), dtype=np.float32)
img_resized = img_resized / 255.
gpu_nms_graph = tf.Graph()

# input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, "./model/yolov3_cpu_nms.pb",
#                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])
input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, model_path,
                                           ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])


with tf.Session(graph=gpu_nms_graph) as sess:
    for i in range(5):
        start = time.time()
        boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        print("=> nms on gpu the number of boxes= %d  time=%.2f ms" % (len(boxes), 1000 * (time.time() - start)))
    print(boxes.shape, scores.shape, labels.shape)
    image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
