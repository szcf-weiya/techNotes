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
