#!/bin/bash
if [ $1 == Fedora -o $1 == Gentoo -o $1 == Redhat ]; then
    echo "The $1 is Redhat Series."
elif [ $1 == Suse -o $1 == Opensuse ]; then
    echo "The $1 is Suse Series."
elif [ $1 == Ubuntu -o $1 == Mint -o $1 == Debian ]; then
    echo "The $1 is Debian Series."
else
    echo "The $1 is Unknown Series."
fi