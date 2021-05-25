#!/bin/bash
Shell=`grep "^$1:" /etc/passwd | cut -d: -f7`
if [[ -z $Shell ]]; then
    echo "No such user"
fi
if [[ $Shell =~ sh$ ]]; then
    echo "Login user"
else
    echo "Not a login user"
fi
