# -*- coding: utf-8 -*-
import os
from PIL import Image
import tensorflow as tf

tagged_img_dir = 'train/tagged_img'

for img_name in os.listdir(tagged_img_dir):
    img_name_stripped = img_name.replace('.png', '')
    tag = img_name_stripped.split('_')[1]
    img = Image.open(os.path.join(tagged_img_dir, img_name)).tobytes()
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))


