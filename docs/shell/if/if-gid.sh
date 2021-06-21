#!/bin/bash
if id $1 &> /dev/null; then
    # not -G
    if [[ `id -u $1` -eq `id -g $1` ]]; then
        echo "gid = uid"
    else
        echo "gid not uid"
    fi
else
    echo "not exist"
fi