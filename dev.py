# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import copy
import numpy as np
import requests
from WHUCaptcha import WHUCaptcha
from tflearn_rewrite import CaptchaModel
import tensorflow as tf

mapping = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])


with tf.Graph().as_default():
    number = '2015302580161'
    password_hash = '95437401ae6fca3869c261a374a8b3e2'
    page_number = 0
    server = 'http://210.42.121.134'
    captcha_url = server + '/servlet/GenImg'
    login_url = server + '/servlet/Login'
    login_success_url = server + '/servlet/../stu/stu_index.jsp'
    model = CaptchaModel.get_model_with_meta_data()
    # while True:
    success_count = 0
    failed_count = 0
    for i in range(500):
        session = requests.session()
        response = session.get(captcha_url)
        if response.status_code == 200:
            with open('captcha.png', 'wb') as f:
                f.write(response.content)
            characters, success = WHUCaptcha.pre_process()
            if success:
                results = model.predict(characters)
                result_str = ''.join(mapping[results.argmax(axis=1)])
                while True:
                    params = {
                        'id': number,
                        'pwd': password_hash,
                        'xdvfb': result_str,
                    }
                    response = session.get(login_url, params=params)
                    if response.url == login_success_url:
                        print result_str
                        print 'auto login success'
                        if result_str.__contains__('i'):
                            raw_input('there is a positive i')
                        break
                        # success_count += 1
                    else:
                        print result_str
                        print 'retry login'
                        result_str = raw_input('input manual captcha:')
                        # failed_count += 1
            pass
    print success_count
    print failed_count





