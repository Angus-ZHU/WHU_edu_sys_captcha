# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import copy
import numpy as np
import requests
from WHUCaptcha import WHUCaptcha


while True:
    captcha_url = 'http://210.42.121.134/servlet/GenImg'
    with open('captcha.png', 'wb') as f:
        f.write(requests.get(captcha_url).content)
    characters, success = WHUCaptcha.pre_process()
    pass





