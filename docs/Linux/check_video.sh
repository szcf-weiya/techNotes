#!/bin/bash
# it would contain the historical tabs
# https://stackoverflow.com/questions/4968271/chrome-on-linux-query-the-browser-to-see-what-tabs-are-open
# n=`strings ~/.config/google-chrome/Default/Sessions/Session_* | sed -nE 's/^([^:]+):\/\/(.*)\/$/\2/p' | grep -v "newtab" | grep -v "new-tab-page" | grep "chinaq.cc\|youtube.com\|bilibili.com" | wc -l`
# n=`strings ~/.config/google-chrome/Default/Sessions/Session_* | sed -nE 's/^([^:]+):\/\/(.*)\/$/\2/p' | grep -v "newtab" | grep -v "new-tab-page" | grep "chinaq.cc\|youtube.com/watch\|bilibili.com/video" | wc -l`
### ok show the current focus tab
# info=`wmctrl -d`
n=`wmctrl -l | grep "bilibili\|中國人線上看\|YouTube\|知乎" | wc -l`
if [ $n -gt 0 ]; then
    notify-send -i /home/weiya/.local/share/icons/warning-icon.svg "不要玩!!!"
fi