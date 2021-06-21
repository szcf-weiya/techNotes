#!/bin/bash
# refer to https://blog.51cto.com/64314491/1629175
sum=0
for I in `seq 1 $#`; do
    sum=$(($sum+$1))
    shift
done
echo $sum