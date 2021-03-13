#!/bin/bash
i=0 # no space before i
i=$((i+1)) # no space before i
# or
i=$(($i+1))
((i++))
((i+=1))
echo $i

a=30
b=10
echo $((a+=b))
echo $((a*=b))
echo $((a-=b))
echo $((a/=b))
echo $((a/=b))
