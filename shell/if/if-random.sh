#!/bin/bash
A=$RANDOM
B=$RANDOM

if [ $A -gt $B ]; then
	echo "Max number is $A."
else
	echo "Min number is $B."
fi
