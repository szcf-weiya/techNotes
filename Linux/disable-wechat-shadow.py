#!/usr/bin/env python3
# refer: https://www.wootec.top/2020/02/16/wine-wechat%E9%98%B4%E5%BD%B1%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/
import time
import os

while True:
    time.sleep(5)
    exist = os.popen("ps -ef | grep WeChat.exe")
    e = exist.readlines()
    if len(e) < 3:
        print(e)
        print("WeChat not started. Exit...")
        exit()
    output = os.popen("wmctrl -l -G -p -x")
    s = output.readlines()
    print(s)
    id = ''
    for item in s:
        if item.find("wechat.exe") != -1:
            id = item.split()[0]
            break
    output.close()
    print(id)
    if id != '':
        #shadow = id[:-4] + "0014"
        shadow = hex(int(id, 16) + 8)
        print(shadow)
        os.system("xdotool windowunmap " + shadow)
    else:
        print("WeChat not display yet.")
