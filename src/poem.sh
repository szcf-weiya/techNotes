#!/bin/bash
txt=$(sed -n "$(($RANDOM % (288-254) + 254))p" /home/weiya/github/cn/_posts/2022-12-31-reading-summary.md)
#n=$(echo $txt | wc -m)
n=${#txt}
if [[ -n ${txt// /} ]]; then
    if [[ $n -gt 11 ]]; then
        notify-send -i /home/weiya/github/techNotes/src/pen-nib-solid.svg -u critical -t 10 ' ' "${txt:1}"
    fi
fi
