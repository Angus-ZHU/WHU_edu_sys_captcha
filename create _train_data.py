# -*- coding: utf-8 -*-
import numpy as np
import cv2
import json
from WHUCaptcha import WHUCaptcha


def get_tag_from_box():
    data = {}

    for line in open('train/train.box').readlines():
        line = line.strip('\n')
        line_data = line.split(' ')
        if line_data[5] not in data:
            data[line_data[5]] = [line_data[0].lower(), ]
        else:
            data[line_data[5]].append(line_data[0].lower())

    normalized_data = {}

    for key, data in data.iteritems():
        if len(data) == 4:
            normalized_data[key] = data

    print len(normalized_data)

    json.dump(normalized_data, open('train\character_tag_lower.json', 'w'))


def split_and_tag():
    i = 0
    tags = json.load(open('train\character_tag_lower.json'))
    for img_number, character_tags in tags.iteritems():
        img = cv2.imread('train/img/train_%s.JPG' % (int(img_number) + 1))
        upper_not_red_bgr = np.array([100, 100, 120])
        lower_not_red_bgr = np.array([0, 0, 0])
        binary_img = cv2.inRange(img, lower_not_red_bgr, upper_not_red_bgr)
        character_list = WHUCaptcha.split_characters(binary_img)
        character_list = WHUCaptcha.normalize_characters(character_list)
        if len(character_list) == 4:
            for index in range(4):
                img = character_list[index]
                tag = character_tags[index]
                cv2.imwrite('train/tagged_img/%s_%s.png' % (i, tag), img)
                i += 1
        else:
            pass

split_and_tag()