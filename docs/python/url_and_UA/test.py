#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 01:49:43 2018

@author: weiya
"""

import requests
import re

class proxyVisit():

    def __init__(self, url):
        self.url = url
        self.proxies = {'http': 'socks5://127.0.0.1:1080','https':'socks5://127.0.0.1:1080'}
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '

           'AppleWebKit/537.36 (KHTML, like Gecko) '

           'Chrome/56.0.2924.87 Safari/537.36'}
        
        self.session = requests.Session()
        self.session.proxies = self.proxies
        self.session.headers = self.headers
        
    def visit(self):
        resp = self.session.get(self.url)
        if resp.status_code == 200:
            print(self.getCurrentIp())
            print("succeed!")
        else:
            print("failed!")
                
    
    def getCurrentIp(self):        
        url = 'http://ip.cn'
        html = self.session.get(url).text
        return(re.search('\d+.\d+.\d+.\d+', html).group(0))

if __name__ == "__main__":
    obj = proxyVisit('https://esl.hohoweiya.xyz')
    for i in range(10):
        obj.visit()
