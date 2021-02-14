#!/bin/bash
# source: https://linuxize.com/post/how-to-compare-strings-in-bash/

# VAR1="Linux"
# VAR2="Linux"
read -p "enter 1st str: " VAR1
read -p "enter 2nd str: " VAR2

if [[ $VAR1 == $VAR2 ]]; then
  echo "equal"
else
  echo "unequal"
fi
