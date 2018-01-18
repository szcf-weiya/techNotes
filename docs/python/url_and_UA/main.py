#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 00:47:24 2018

@author: weiya
"""

import requests
#from bs4 import BeautifulSoup as bs

class proxyVisit():
    def __init__(self):
        self.url = 'https://blog.hohoweiya.xyz'
        self.proxies = {'http': 'socks5://127.0.0.1:1080','https':'socks5://127.0.0.1:1080'}
        
    def visit(self):
        response = requests.get(self.url, proxies=self.proxies)
        print(response.content.decode('utf8'))

if __name__ == "__main__":
    obj = proxyVisit()
    obj.visit()