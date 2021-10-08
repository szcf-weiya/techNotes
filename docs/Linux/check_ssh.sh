#!/bin/bash
ssh -q -o BatchMode=yes $1 exit
if [ $? != '0' ]; then
    #echo "broken connection"
    notify-send "$1" "broken connection"
fi
