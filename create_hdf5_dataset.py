# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import h5py

tagged_img_dir = 'train/tagged_img'
mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

label_data_set = []
img_data_set = []
for img_name in os.listdir(tagged_img_dir):
    img_name_stripped = img_name.replace('.png', '')
    tag = img_name_stripped.split('_')[1]
    img = cv2.imread(os.path.join(tagged_img_dir, img_name))
    upper_not_red_bgr = np.array([255, 255, 255])
    lower_not_red_bgr = np.array([100, 100, 100])
    binary_img = cv2.inRange(img, lower_not_red_bgr, upper_not_red_bgr)
    # tag_array = np.zeros(36, dtype=np.int8)
    # tag_array[mapping.index(tag)] = 1
    label_data_set.append(tag)
    img_data_set.append(binary_img)

with h5py.File('train/train_letter_tag.h5', 'w') as f:
    f.create_dataset('img', data=img_data_set)
    f.create_dataset('label', data=label_data_set)