#!/bin/bash
sum=0
for I in `seq 1 $#`; do
    sum=$(($sum+$I))
done
echo $sum