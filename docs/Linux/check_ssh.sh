#!/bin/bash
ssh -t $2 ssh -q -o BatchMode=yes $1 exit
if [ $? != '0' ]; then
    #echo "broken connection"
    notify-send "$1" "broken connection"
fi
