# Kazam

Ubuntu 下 kazam 录屏 没声音解决方案

http://www.cnblogs.com/xn--gzr/p/6195317.html


# bluebooth

[set bluebooth](http://dz.sdut.edu.cn/blog/subaochen/2017/02/ubuntu%E4%B8%8B%E9%80%9A%E8%BF%87%E8%93%9D%E7%89%99%E6%8E%A5%E6%94%B6%E6%89%8B%E6%9C%BA%E5%8F%91%E9%80%81%E7%9A%84%E6%96%87%E4%BB%B6/)

# sogou input problem

[input problem](http://pinyin.sogou.com/bbs/forum.php?mod=viewthread&tid=2681098&extra=page%3D1)

# install win on ubuntu
http://www.linuxdeveloper.space/install-windows-after-linux/

# fix locale issue
https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue

阿里云的服务器，最好default为'zh_CN.UTF-8'

# add user
http://blog.csdn.net/linuxdriverdeveloper/article/details/7427672

# sudo

http://blog.csdn.net/ichuzhen/article/details/8241847

# 初始化服务器

1. 新建用户，sudo
2. 添加sources.list,gpg
3. 安装R
4. 安装Rstudioserver（成功！！！哎。。搞了一下午就是因为上午莫名其妙更新了Ubuntu，不要手贱！！）

# 终端分屏
[tmux](http://blog.csdn.net/u010454729/article/details/49496381)

# 缺少动态链接库

在服务器上使用gsl报缺少动态链接库的错误
解决方案
[3种方法](http://www.cnblogs.com/smartvessel/archive/2011/01/21/1940868.html)

另参考
http://blog.csdn.net/wangeen/article/details/8159500

```
sudo vim /etc/ld.so.conf
```

添加

```
/where/is/the/lib/
```


# Ubuntu 下对文本文件每行行首进行追加、替换

[sed](http://blog.csdn.net/u010555688/article/details/48416765)


# makefile

https://my.oschina.net/u/1413984/blog/199029

$@: 目标文件
$^: 所有的依赖文件
$<: 第一个依赖文件


# atom 自动更新
[atom](https://launchpad.net/~webupd8team/+archive/ubuntu/atom/)

```
sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
```

# 配置xterm

## 中文字体的问题

查看本机安装的中文字体
```
fc-list :lang=zh
```

选出一个字体的名称写进配置文件中，如
```
xterm*faceNameDoublesize: YaHei Consolas Hybrid
```

参考

1. http://forum.ubuntu.org.cn/viewtopic.php?t=143221

## could not get lock /var/lib/dpkg/lock -open

```
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

如果不行，重启。

## 阿里云服务器virtual memory exhausted: Cannot allocate memory

http://www.bubuko.com/infodetail-1319039.html
```
dd if=/dev/zero of=/swap bs=1024 count=1M    #创建一个大小为1G的文件/swap
mkswap /swap                                                 #将/swap作为swap空间
swapon /swap                                                  #enable /swap file  for paging and swapping
echo "/swap swap swap sw 0 0" >> /etc/fstab    #Enable swap on boot, 开机后自动生效
```

## 编译安装gcc-4.6.2

1. https://gcc.gnu.org/faq.html#multiple
2. https://askubuntu.com/questions/313288/how-to-use-multiple-instances-of-gcc
3. http://www.tellurian.com.au/whitepapers/multiplegcc.php
4. https://stackoverflow.com/questions/9450394/how-to-install-gcc-piece-by-piece-with-gmp-mpfr-mpc-elf-without-shared-libra

## 更新rstudio 后闪退
1. 安装rstudio应该采用

```
sudo apt-get install gdebi-core
wget https://download1.rstudio.org/rstudio-1.0.44-amd64.deb
sudo gdebi rstudio-1.0.44-amd64.deb
```

而非
```
sudo dpkg -i
```

另外，如果不行，删除后再装
```
sudo apt-get remove rstudio
```

## gcc版本
1. https://codeyarns.com/2015/02/26/how-to-switch-gcc-version-using-update-alternatives/

## terminator设置
1. hostname的颜色
https://stackoverflow.com/questions/40077907/is-it-possible-to-customize-terminators-prompt-hostname-userdomain-colors
直接打开bashrc里面下一行的注释
```
#force_color_prompt=yes
```
2. 颜色背景色等，直接右键设置，右键设置完成之后便有了一个config文件.


## 试图在Ubuntu，rvpn回去

参考的资料有
1. [vpn-pptp-in-ubuntu-16-04-not-working](https://askubuntu.com/questions/891393/vpn-pptp-in-ubuntu-16-04-not-working)


## flatten pdf file

参考[is-there-a-way-to-flatten-a-pdf-image-from-the-command-line](https://unix.stackexchange.com/questions/162922/is-there-a-way-to-flatten-a-pdf-image-from-the-command-line)

```
pdf2ps orig.pdf - | ps2pdf - flattened.pdf
```

## Linux 杀进程

参考[linux下杀死进程（kill）的N种方法](http://blog.csdn.net/andy572633/article/details/7211546)

```
ps -ef | grep R
kill -s 9 ...
```

## 合并jpg到pdf

参考[convert images to pdf: How to make PDF Pages same size](https://unix.stackexchange.com/questions/20026/convert-images-to-pdf-how-to-make-pdf-pages-same-size)

直接采用

```
pdftk A.pdf B.pdf cat output merge.pdf
```

得到的pdf中页面大小不一致，于是采用下面的命令

```
convert a.png b.png -compress jpeg -resize 1240x1753 \
                      -extent 1240x1753 -gravity center \
                      -units PixelsPerInch -density 150x150 multipage.pdf
```

注意重点是`-density 150x150`，若去掉这个选项，则还是得不到相同页面大小的文件。

另外，上述命令是对于`.png`而言的，完全可以换成`.jpg`。

同时，注意`1240x1753`中间是字母`x`.

## install typora on Linux

参考[Install Typora on Linux](http://support.typora.io/Typora-on-Linux/)

## Rstudio 不能切换中文输入（fctix）

参考[Rstudio 不能切换中文输入（fctix）](http://blog.csdn.net/qq_27755195/article/details/51002620)

[Ubuntu 16.04 + Fcitx + RStudio 1.0で日本語を入力する方法](http://blog.goo.ne.jp/ikunya/e/8508d21055503d0560efc245aa787831)

## 配置jdk

参考[Ubuntu14.04安装JDK与配置环境变量](https://jingyan.baidu.com/article/647f0115bb26817f2048a871.html)

## 缩小图像的大小

```
convert -resize 1024x
```

或者

```
convert -quality 50%
```

具体参考[How can I compress images?](https://askubuntu.com/questions/781497/how-can-i-compress-images)

## compile FileZilla

refer to [Client Compile](https://wiki.filezilla-project.org/Client_Compile)

download latest libfilezilla from https://lib.filezilla-project.org/download.php

add wxWidget's repository according to http://codelite.org/LiteEditor/WxWidgets31Binaries#toc2

pay attention to the version, NOT 3.1.0.

http://codelite.org/LiteEditor/WxWidgets30Binaries

require libgnutls 3.4.15 or greater, download from  https://gnutls.org/

require sqlite3.h
```
sudo apt-get install libsqlite3-dev
```

## convert 参数

pdf 转为 jpg
 `-quality 100` 控制质量
 `-density 600x600` 控制分辨率

 并注意参数放置文件的前面

## linux 三款命令行浏览器

1. w3m
2. links
3. lynx

refer to http://www.laozuo.org/8178.html

## 修改文件权限

采用`ls -l` 便可以查看文件(夹)权限，比如

```bash
-rw-rw-r--  1 weiya weiya    137969 3月   8  2017 font.txt
-rw-r--r--  1 root  root      35792 12月 26 23:50 geckodriver.log
-rw-r--r--  1 root  root     327350 12月 27 01:38 ghostdriver.log
```
7列的含义分别是（参考http://blog.csdn.net/jenminzhang/article/details/9816853）

1. 文件类型和文件权限
2. 文件链接个数
3. 文件所有者
4. 文件所在群组
5. 文件长度
6. 时间
7. 文件名称

采用chmod修改权限（参考http://www.linuxidc.com/Linux/2015-03/114695.htm），如
```bash
chmod -R 700 Document/
```

其中`-R`递归

采用chown改变所有者，比如
```bash
chown -R username:users Document/
```

## 腾讯云服务器nginx failed

原因：80端口被占用
解决方法：kill掉占用80端口的

```
sudo fuser -k 80/tcp
```

重启

```
sudo /etc/init.d/nginx restart
```

# 文件重命名

参考[Ubuntu中rename命令和批量重命名](http://www.linuxidc.com/Linux/2016-11/137041.htm)

```bash
rename -n 's/Sam3/Stm32/' *.nc　　/*确认需要重命名的文件*/
rename -v 's/Sam3/Stm32/' *.nc　　/*执行修改，并列出已重命名的文件*/
```
