#!/bin/bash
if [[ $[$RANDOM % 2] -eq 1 ]]; then
    txt=$(sed -n "$(($RANDOM % (288-254) + 254))p" /home/weiya/github/cn/_posts/2022-12-31-reading-summary.md)
    #n=$(echo $txt | wc -m)
    n=${#txt}
    if [[ -n ${txt// /} ]]; then
        if [[ $n -gt 11 ]]; then
            notify-send -i /home/weiya/github/techNotes/src/pen-nib-solid.svg -u critical -t 10 ' ' "${txt:1}"
        fi
    fi
else
    nline=$(awk -F, '$13 !~ /^\r$/{print $13}' ~/PGitHub/Su-Shi/Su-shi.csv | wc -l)
    rand=$(shuf -i 1-$nline -n 1)
    txt=$(awk -F, '$13 !~ /^\r$/{print $13}' ~/PGitHub/Su-Shi/Su-shi.csv | awk -v line=$rand 'NR==line {print; exit}')
    notify-send -i /home/weiya/github/techNotes/src/pen-nib-solid.svg -u critical -t 10 ' ' "${txt}"
fi