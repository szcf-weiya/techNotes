#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 02:02:33 2018

@author: weiya
"""

import requests

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '

           'AppleWebKit/537.36 (KHTML, like Gecko) '

           'Chrome/56.0.2924.87 Safari/537.36'}

proxies = {'http': 'socks5://127.0.0.1:1080','https':'socks5://127.0.0.1:1080'}

url = 'http://ip.cn'

response = requests.get(url, proxies=proxies)

print(response.content)