import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests
while True:
    captcha_url = 'http://210.42.121.241/servlet/GenImg'
    with open('captcha.png', 'wb') as f:
        f.write(requests.get(captcha_url).content)
    # 注意opencv是
    bgr_img = cv2.imread('captcha.png')
    upper_red_bgr = np.array([255, 255, 130])
    lower_red_bgr = np.array([0, 0, 0])
    bgr_filter_res = cv2.inRange(bgr_img, lower_red_bgr, upper_red_bgr)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    upper_red_hsv = np.array([180, 255, 255])
    lower_red_hsv = np.array([0, 170, 0])
    hsv_filter_res = cv2.inRange(hsv_img, lower_red_hsv, upper_red_hsv)
    mix = hsv_filter_res - bgr_filter_res

    (b, g, r) = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    plt.subplot(511)
    plt.imshow(rgb_img)
    plt.subplot(512)
    plt.imshow(bgr_filter_res, 'gray_r')
    plt.subplot(513)
    plt.imshow(hsv_img)
    plt.subplot(514)
    plt.imshow(hsv_filter_res, 'gray_r')
    plt.subplot(515)
    plt.imshow(mix, 'gray_r')
    mng = plt.get_current_fig_manager()
    mng.resize(*(500, 800))
    plt.show()
