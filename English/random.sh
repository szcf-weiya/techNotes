#!/bin/bash
txt=$(sed -n "$(($RANDOM % $(wc -l < index.md) + 1))p" index.md)
# https://stackoverflow.com/questions/13509508/check-if-string-is-neither-empty-nor-space-in-shell-script
if [ -n ${txt// /} ]; then
    if [ ${txt:0:1} != '#' ]; then
        notify-send -i $(pwd)/english-speaking-icon.svg -u critical -t 10 ' ' "${txt:1}"
    fi
fi
