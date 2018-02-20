# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:11:22 2018

@author: weiya
"""

import requests

url = 'https://sms-api.upyun.com/api/messages'
postdata = {
        'mobile' : '17816859236',
        'template_id': 1,
        'vars' : '高悦'
        }
headers = {
        'Content-type' : 'application/x-www-form-urlencoded',
        'Authorization': 'IVxYP6t64gMqrfPOlcMXR9HvhJb8ji'
        }

req = requests.post(url, headers = headers, data=postdata)
print(req.text)