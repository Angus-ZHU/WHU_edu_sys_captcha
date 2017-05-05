# -*- coding: utf-8 -*-
import requests
from WHUCaptcha import WHUCaptcha

while True:
    captcha_url = 'http://210.42.121.241/servlet/GenImg'
    with open('captcha.png', 'wb') as f:
        f.write(requests.get(captcha_url).content)
    mix = WHUCaptcha.pre_process()



