# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import copy
import numpy as np
import requests
from WHUCaptcha import WHUCaptcha




while True:
    captcha_url = 'http://210.42.121.241/servlet/GenImg'
    with open('captcha.png', 'wb') as f:
        f.write(requests.get(captcha_url).content)
    mix = WHUCaptcha.pre_process()


    # 用opencv自带的找轮廓然后框出来的效果不是很好
    # ij字符上面那个点到是可以合并，
    # 但是的粘连，还有降噪的时候造成的字符断开很致命
    # plt.subplot(211)
    # plt.imshow(mix, 'gray_r')
    # plt.show()
    # image = copy.deepcopy(mix)
    # image_middle = copy.deepcopy(mix)
    # image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # rects = []
    # for contour in contours:
    #     if cv2.contourArea(contour) < 2:
    #         continue
    #     rect = cv2.boundingRect(contour)
    #     # box = cv2.boxPoints(rect)
    #     # box = np.int8(box)
    #     rects.append(rect)
    # rects.sort()
    # for rect in rects:
    #     cv2.rectangle(image_middle, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 100, 0)
    # if len(rects) == 4:
    #     pass
    # else:
    #     new_rects = []
    #     for index, rect in enumerate(rects):
    #         if index < len(rects)-1 and rects[index+1][0] - rect[0] <= 3:
    #             new_rect = join_rect(rects[index+1], rect)
    #             new_rects.append(new_rect)
    #         else:
    #             new_rects.append(rect)
    #     rects = new_rects
    # for rect in rects:
    #     cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 100, 0)
    # plt.subplot(311)
    # plt.imshow(mix, 'gray_r')
    # plt.subplot(312)
    # plt.imshow(image_middle, 'gray_r')
    # plt.subplot(313)
    # plt.imshow(image, 'gray_r')
    # plt.show()
    # pass



