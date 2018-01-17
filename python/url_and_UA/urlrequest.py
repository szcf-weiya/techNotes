#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:13:14 2018

@author: weiya
"""

from urllib import request as urlrequest

ipList = [
    '203.125.234.1',   '220.181.7.1',     '123.125.66.1',
    '123.125.71.1',    '119.63.192.1',    '119.63.193.1',
    '119.63.194.1',    '119.63.195.1',    '119.63.196.1',
    '119.63.197.1',    '119.63.198.1',    '119.63.199.1',
    '180.76.5.1',      '202.108.249.185', '202.108.249.177',
    '202.108.249.182', '202.108.249.184', '202.108.249.189',
    '61.135.146.200',  '61.135.145.221',  '61.135.145.207',
    '202.108.250.196', '68.170.119.76',   '207.46.199.52',
]

url = 'http://www.httpbin.org/ip'

def visit():    
    for i in range(len(ipList)):
        proxy_host = ipList[i]
    
        req = urlrequest.Request(url)
        req.set_proxy(proxy_host, 'http')
        try:    
            response = urlrequest.urlopen(req)
            print(response.read().decode('utf8'))    
        except:
            print("failed!")
            continue
        
        
if __name__ == "__main__":
    visit()
