#!/bin/bash
if grep "^$" $1 &> /dev/null; then
    echo "there are `grep "^$" $1 | wc -l` empty lines"
else
    echo "no empty lines"
fi