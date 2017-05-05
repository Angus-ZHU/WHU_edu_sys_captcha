# -*- coding: utf-8 -*-
import cv2
import numpy as np


class WHUCaptcha(object):

    @staticmethod
    def pre_process(img_dir='captcha.png'):
        # 注意opencv是用bgr而不是rgb打开图片的
        bgr_img = cv2.imread(img_dir)
        # bgr筛选出r值较低的噪点
        # 因为存在r值低但是饱和度高的噪点会混入后续hsv筛出的字符中去
        upper_red_bgr = np.array([255, 255, 130])
        lower_red_bgr = np.array([0, 0, 0])
        bgr_filter_res = cv2.inRange(bgr_img, lower_red_bgr, upper_red_bgr)
        # hsv筛出字符的大致轮廓
        # 因为字符的饱和度显著高于其他色块
        # opencv不知为何hsv的取值有些奇怪不是[360, 100, 100]而是[180, 255, 255]
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        upper_red_hsv = np.array([180, 255, 255])
        lower_red_hsv = np.array([0, 170, 0])
        hsv_filter_res = cv2.inRange(hsv_img, lower_red_hsv, upper_red_hsv)
        # 将两次的筛选合并输出最后的二值化字符图像
        mix = hsv_filter_res - bgr_filter_res
        # # matplot可视化中间处理过程
        # # 可以参见example/pre_process*.png
        # from matplotlib import pyplot as plt
        # (b, g, r) = cv2.split(bgr_img)
        # rgb_img = cv2.merge([r, g, b])
        # plt.subplot(511)
        # plt.imshow(rgb_img)
        # plt.subplot(512)
        # plt.imshow(bgr_filter_res, 'gray_r')
        # plt.subplot(513)
        # plt.imshow(hsv_img)
        # plt.subplot(514)
        # plt.imshow(hsv_filter_res, 'gray_r')
        # plt.subplot(515)
        # plt.imshow(mix, 'gray_r')
        # mng = plt.get_current_fig_manager()
        # mng.resize(*(500, 800))
        # plt.show()
        return mix
