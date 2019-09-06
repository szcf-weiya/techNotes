#!/bin/bash

# ./install-java-8.sh 221

# refer to https://stackoverflow.com/questions/13181725/append-file-contents-to-the-bottom-of-existing-file-in-bash
echo "PATH=\$PATH:/usr/lib/jvm/jdk1.8.0_$1/bin:/usr/lib/jvm/jdk1.8.0_$1/db/bin:/usr/lib/jvm/jdk1.8.0_$1/jre/bin" >> ~/.bashrc

echo "J2SDKDIR=/usr/lib/jvm/jdk1.8.0_$1" >> ~/.bashrc
echo "J2REDIR=/usr/lib/jvm/jdk1.8.0_$1/jre" >> ~/.bashrc
echo "JAVA_HOME=/usr/lib/jvm/jdk1.8.0_$1" >> ~/.bashrc
echo "DERBY_HOME=/usr/lib/jvm/jdk1.8.0_$1/db" >> ~/.bashrc

sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk1.8.0_$1/bin/java" 0
sudo update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk1.8.0_$1/bin/javac" 0
sudo update-alternatives --set java /usr/lib/jvm/jdk1.8.0_$1/bin/java
sudo update-alternatives --set javac /usr/lib/jvm/jdk1.8.0_$1/bin/javac
