#!/bin/bash
ssh -t $2 ssh -q -o BatchMode=yes $1 exit
if [ $? != '0' ]; then
    #echo "broken connection"
    notify-send -i /home/weiya/.local/share/icons/unlink-solid.svg "$1" "broken connection"
fi
