#!/bin/bash
# refer to https://blog.51cto.com/64314491/1629175
if id $1; then
	echo "$1 exists"
else
	echo "$1 is not exists"
fi

