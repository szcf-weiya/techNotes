# Linux笔记

## install win on ubuntu

参考[http://www.linuxdeveloper.space/install-windows-after-linux/](http://www.linuxdeveloper.space/install-windows-after-linux/)

## fix locale issue

参考[https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue](https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue)

阿里云的服务器，最好default为'zh_CN.UTF-8'

## add user

参考[https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04](http://blog.csdn.net/linuxdriverdeveloper/article/details/7427672)

```bash
useradd -m -s /bin/bash userName
passwd userName
```

增加sudo权限

```bash
sudoedit /etc/sudoers
// 在配置文件中找到如下位置，并添加userName1那一行。
## Allow root to run any commands anywhere
root    ALL=(ALL)       ALL
userName1 ALL=(ALL)       NOPASSWD:ALL
userName2 ALL=(ALL)       ALL
```

## unable to resolve host

参考[http://blog.csdn.net/ichuzhen/article/details/8241847](http://blog.csdn.net/ichuzhen/article/details/8241847)

## 初始化服务器

1. 新建用户，sudo
2. 添加sources.list,gpg
3. 安装R
4. 安装Rstudioserver（成功！！！哎。。搞了一下午就是因为上午莫名其妙更新了Ubuntu，不要手贱！！）

## 终端分屏

参考 [linux 工具——终端分屏与vim分屏](http://blog.csdn.net/u010454729/article/details/49496381)

还可以切换后台运行，在服务器上操作特别方便。

常用操作

```bash
# new a shell
tmux
# new a shell with name
tmux new -s NAME
# view all shell
tmux ls
# go back
tmux attach-session -t [NUM]
# simplify
tmux attach -t [NUM]
# more simplify
tmux a -t [NUM]
# via name
tmux a -t NAME
# complete reset: https://stackoverflow.com/questions/38295615/complete-tmux-reset
tmux kill-server
```

refer to 
- [How do I access tmux session after I leave it?](https://askubuntu.com/questions/824496/how-do-i-access-tmux-session-after-i-leave-it)
- [Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)
- [tmux cheatsheet](https://gist.github.com/henrik/1967800)
## 缺少动态链接库

在服务器上使用gsl报缺少动态链接库的错误
解决方案
[3种方法](http://www.cnblogs.com/smartvessel/archive/2011/01/21/1940868.html)

另参考
[http://blog.csdn.net/wangeen/article/details/8159500](http://blog.csdn.net/wangeen/article/details/8159500)

```
sudo vim /etc/ld.so.conf
```

添加

```
/where/is/the/lib/
```


## Vim 对每行行首进行追加、替换

按住 v 或者 V 选定需要追加的行，然后再进入 `:` 模式，输入正常的 `sed` 命令，如

```bash
s/^/#/g
```

参考 [Ubuntu 下对文本文件每行行首进行追加、替换](http://blog.csdn.net/u010555688/article/details/48416765)


## Make

[https://my.oschina.net/u/1413984/blog/199029](https://my.oschina.net/u/1413984/blog/199029)

- `$@`: 目标文件，比如 [gsl_lm/ols/Makefile](https://github.com/szcf-weiya/gsl_lm/blob/86d8c4846ed56a27ad8a9f35d9f1229fab704912/ols/Makefile#L22)
- `$^`: 所有的依赖文件，比如 [G-squared/src/Makefile](https://github.com/szcf-weiya/G-squared/blob/4f70c3f735e4241f7ba33986c9b6a53fdd0dc6ea/src/Makefile#L9-L21)
- `$<`: 第一个依赖文件

and [Makefile 经典教程(掌握这些足够)](http://blog.csdn.net/ruglcc/article/details/7814546/)


## 配置xterm的中文字体的问题

查看本机安装的中文字体
```
fc-list :lang=zh
```

选出一个字体的名称写进配置文件中，如
```
xterm*faceNameDoublesize: YaHei Consolas Hybrid
```

参考

1. [http://forum.ubuntu.org.cn/viewtopic.php?t=143221](http://forum.ubuntu.org.cn/viewtopic.php?t=143221)

## could not get lock /var/lib/dpkg/lock -open

```
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

如果不行，重启。

## 阿里云服务器virtual memory exhausted: Cannot allocate memory

[http://www.bubuko.com/infodetail-1319039.html](http://www.bubuko.com/infodetail-1319039.html)
```
##创建一个大小为1G的文件/swap
dd if=/dev/zero of=/swap bs=1024 count=1M
##将/swap作为swap空间
mkswap /swap
##enable /swap file  for paging and swapping
swapon /swap
##Enable swap on boot, 开机后自动生效
echo "/swap swap swap sw 0 0" >> /etc/fstab
```


## gcc版本

可以通过 `update-alternatives` 进行切换，但注意要提前安装 `install` alternatives，这里的 install 不是下载源码安装，而是将系统中已有的不同版本的 gcc 安装到 alternatives 中。比如当前我电脑的 gcc --version 是 7.5.0，但是仍有 `gcc-5`, `gcc-4.8` 等命令，不过这些并不在 alternatives 中，因为如果直接运行 

```bash
$ sudo update-alternatives --config gcc
update-alternatives: error: no alternatives for gcc
```

所以可以按照 [How to switch GCC version using update-alternatives](https://codeyarns.github.io/tech/2015-02-26-how-to-switch-gcc-version-using-update-alternatives.html) 先

```bash
sudo update-alternatives --install ....
```

然后再 config.


## terminator设置
1. hostname的颜色
https://stackoverflow.com/questions/40077907/is-it-possible-to-customize-terminators-prompt-hostname-userdomain-colors
直接打开bashrc里面下一行的注释
```
##force_color_prompt=yes
```
2. 颜色背景色等，直接右键设置，右键设置完成之后便有了一个config文件.


## flatten pdf file

参考[is-there-a-way-to-flatten-a-pdf-image-from-the-command-line](https://unix.stackexchange.com/questions/162922/is-there-a-way-to-flatten-a-pdf-image-from-the-command-line)

```
pdf2ps orig.pdf - | ps2pdf - flattened.pdf
```

### Linux 杀进程

参考[linux下杀死进程（kill）的N种方法](http://blog.csdn.net/andy572633/article/details/7211546)

```bash
ps -ef | grep R
kill -s 9 ...
```

其中 `ps -ef` 输出格式为

```bash
$ ps -ef | head -2
UID        PID  PPID  C STIME TTY          TIME CMD
root         1     0  0 09:15 ?        00:00:44 /sbin/init splash
```

每一列的含义可以在 `man ps` 中的 `STANDARD FORMAT SPECIFIERS` 小节中找到，具体地，

- `UID`: same with EUID, effective user ID (alias uid).
- `PID`: a number representing the process ID (alias tgid).
- `PPID`: parent process ID.
- `C`: processor utilization. Currently, this is the integer value of the percent usage over the lifetime of the process.  (see %cpu).
- `STIME`: same with `START`, starting time or date of the process.  Only the year will be displayed if the process was not started the same year ps was invoked, or "MmmDD" if it was not started the same day, or "HH:MM" otherwise.  See also bsdstart, start, lstart, and stime.
- `TTY`: controlling tty (terminal).  (alias tt, tty).
- `TIME`: cumulative CPU time, "[DD-]HH:MM:SS" format.  (alias cputime).
- `CMD`: see args.  (alias args, command).

### 合并jpg到pdf

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

download latest libfilezilla from [https://lib.filezilla-project.org/download.php](https://lib.filezilla-project.org/download.php)

add wxWidget's repository according to [http://codelite.org/LiteEditor/WxWidgets31Binaries#toc2](http://codelite.org/LiteEditor/WxWidgets31Binaries#toc2)

pay attention to the version, NOT 3.1.0.

[http://codelite.org/LiteEditor/WxWidgets30Binaries](http://codelite.org/LiteEditor/WxWidgets30Binaries)

require libgnutls 3.4.15 or greater, download from  [https://gnutls.org/](https://gnutls.org/)

require sqlite3.h
```
sudo apt-get install libsqlite3-dev
```

## convert 参数

pdf 转为 jpg
 `-quality 100` 控制质量
 `-density 600x600` 控制分辨率

 并注意参数放置文件的前面

pdf 转 png 更好的命令是 `pdftoppm`，参考 [How to convert PDF to Image?](https://askubuntu.com/questions/50170/how-to-convert-pdf-to-image)

```bash
pdftoppm alg.pdf alg -png -singlefile
```

图片质量比 `convert` 好很多！！

## linux 三款命令行浏览器

1. w3m
2. links
3. lynx

refer to [http://www.laozuo.org/8178.html](http://www.laozuo.org/8178.html)

## 修改文件权限

采用`ls -l` 便可以查看文件(夹)权限，比如

```bash
-rw-rw-r--  1 weiya weiya    137969 3月   8  2017 font.txt
-rw-r--r--  1 root  root      35792 12月 26 23:50 geckodriver.log
-rw-r--r--  1 root  root     327350 12月 27 01:38 ghostdriver.log
```
7列的含义分别是（参考[http://blog.csdn.net/jenminzhang/article/details/9816853](http://blog.csdn.net/jenminzhang/article/details/9816853)）

1. 文件类型和文件权限
2. 文件链接个数
3. 文件所有者
4. 文件所在群组
5. 文件长度
6. 时间
7. 文件名称

采用chmod修改权限（参考[http://www.linuxidc.com/Linux/2015-03/114695.htm](http://www.linuxidc.com/Linux/2015-03/114695.htm)），如
```bash
chmod -R 700 Document/
```

其中`-R`递归

采用chown改变所有者，比如
```bash
chown -R username:users Document/
```

`chmod g+s .` 会使得当前文件夹 `.` 中所有新建文件或文件夹都继承 `.` 的 group，而不是创建者所属的 group，所以这一般配合 `chgrp` 使用。参考 ['chmod g+s' command](https://unix.stackexchange.com/questions/182212/chmod-gs-command)


## 文件重命名

参考[Ubuntu中rename命令和批量重命名](http://www.linuxidc.com/Linux/2016-11/137041.htm)

```bash
rename -n 's/Sam3/Stm32/' *.nc　　/*确认需要重命名的文件*/
rename -v 's/Sam3/Stm32/' *.nc　　/*执行修改，并列出已重命名的文件*/
```

## 关闭screen

参考[https://stackoverflow.com/questions/1509677/kill-detached-screen-session](https://stackoverflow.com/questions/1509677/kill-detached-screen-session)

```bash
screen -list #或screen -r
screen -r [pid] # 进入
### ctrl+A, 然后输入":quit"
```

更多用法详见 [linux screen 命令详解](https://www.cnblogs.com/mchina/archive/2013/01/30/2880680.html)


## Solution: Client with the currently selected authenticator does not support any combination of challenges that will satisfy the CA
参考
https://community.letsencrypt.org/t/solution-client-with-the-currently-selected-authenticator-does-not-support-any-combination-of-challenges-that-will-satisfy-the-ca/49983

## cairo图形库环境搭建

参考[ubuntu Cairo图形库 环境搭建](http://blog.csdn.net/zh19921107/article/details/45094759)

## circos

介绍见[DOWNLOAD CIRCOS, TUTORIALS AND TOOLS](http://circos.ca/software/download/tutorials/)

[Install circos on ubuntu 14.04 LTS](https://gist.github.com/dyndna/18bb71494e021f672510)

## shell 提取文件名和目录名

[shell 提取文件名和目录名](http://blog.csdn.net/universe_hao/article/details/52640321)

## 几种方法来实现scp拷贝时无需输入密码

[几种方法来实现scp拷贝时无需输入密码](http://blog.csdn.net/nfer_zhuang/article/details/42646849)


## time命令中的real,user以及sys时间

[time命令中的real,user以及sys时间](http://blog.chinaunix.net/uid-23177306-id-2531034.html)

## control android phone by PC's mouse and keyboard

[How to Control Your Android Using Your Computer’s Mouse and Keyboard](https://www.makeuseof.com/tag/control-android-using-computers-mouse-keyboard/)

## fix my locale issue

[How do I fix my locale issue?](https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue)

## 解决Unable to load native-hadoop library for your platform

参考[解决Unable to load native-hadoop library for your platform](http://blog.csdn.net/succeedloveaaaa/article/details/48596857)

## Vultr配置shadowsocks

按照之前的配置方法，不可用，于是参考[轻松在 VPS 搭建 Shadowsocks 翻墙](https://www.diycode.cc/topics/738)进行配置。

## CentOS7搭建Apache

参考资料

1. [How To Install Linux, Apache, MySQL, PHP (LAMP) stack On CentOS 7](https://www.digitalocean.com/community/tutorials/how-to-install-linux-apache-mysql-php-lamp-stack-on-centos-7)
2. [CentOS 7.2 利用yum安装配置Apache2.4多虚拟主机](http://www.linuxidc.com/Linux/2017-10/147667.htm)

按照第一个链接的指示，并不能成功访问。于是尝试参考第二个链接修改配置文件。

未果，结果按照cy的建议，释放掉了这个服务器。

## 命令最后的&

参考[What does “&” at the end of a linux command mean?](https://stackoverflow.com/questions/13338870/what-does-at-the-end-of-a-linux-command-mean)

表示在后台运行。

## crontab定时任务

`* */1 * * * *` 表现为每分钟执行，但是本来第 1 列应当为分钟，而第 2 列为小时，这样使用对用法理解错误，而且改成 `* * */1 * * *` 仍然表现为每分钟。试图

```bash
sudo service cron restart
# or
sudo service cron reload
```

都失败了。所以还是理解出现了偏差，

参考[Linux 设置定时任务crontab命令](https://www.cnblogs.com/zoulongbin/p/6187238.html) 和 [关于定时执行任务：Crontab的20个例子](https://www.jianshu.com/p/d93e2b177814)

## ubuntu 连接 sftp 服务器

参考[Use “Connect to Server” to connect to SFTP](https://askubuntu.com/questions/349873/use-connect-to-server-to-connect-to-sftp)

## 视频旋转

参考[How can I rotate a video?](https://askubuntu.com/questions/83711/how-can-i-rotate-a-video)

直接用

```bash
ffmpeg -i in.mov -vf "transpose=1" out.mov
```

然后报错 [“The encoder 'aac' is experimental but experimental codecs are not enabled”]((https://stackoverflow.com/questions/32931685/the-encoder-aac-is-experimental-but-experimental-codecs-are-not-enabled)) 

注意添加 `-strict -2` 要注意放置位置，一开始直接在上述命令后面加入，但失败，应该写成


```bash
ffmpeg -i in.mov -vf "transpose=1" -strict -2 out.mov
```

## Ubuntu的回收站

参考 [https://blog.csdn.net/DSLZTX/article/details/46685959](https://blog.csdn.net/DSLZTX/article/details/46685959)

## 输出到 log 文件

参考[How do I save terminal output to a file?](https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file)

发现一件很迷的事情，要加上 `-u` 才能实现实时查看输出。

参考

1. [Python: significance of -u option?](https://stackoverflow.com/questions/14258500/python-significance-of-u-option)
2. [后台运行python程序并标准输出到文件](http://www.cnblogs.com/qlshine/p/5926743.html)

## useful commands

1. `cd "$(dirname "$0")"`: [cd current directory](https://stackoverflow.com/questions/3349105/how-to-set-current-working-directory-to-the-directory-of-the-script)
2. `mkdir -p`: [mkdir only if a dir does not already exist?](https://stackoverflow.com/questions/793858/how-to-mkdir-only-if-a-dir-does-not-already-exist)

## xargs 命令

1. [xargs命令_Linux xargs 命令用法详解：给其他命令传递参数的一个过滤器](http://man.linuxde.net/xargs)

## Unable to lock the administration directory (/var/lib/dpkg/) is another process using it?

[Unable to lock the administration directory (/var/lib/dpkg/) is another process using it?](https://askubuntu.com/questions/15433/unable-to-lock-the-administration-directory-var-lib-dpkg-is-another-process)

## redirection

Note that 

>  when you put & the output - both stdout and stderr - will still be printed onto the screen.

> If you do not want to see any output on the screen, redirect both stdout and stderr to a file by:

```bash
myscript > ~/myscript.log 2>&1 &
```
or just 

```bash
myscript > /dev/null 2>&1 &
```

refer to [Why can I see the output of background processes?](https://askubuntu.com/questions/662817/why-can-i-see-the-output-of-background-processes)

[Formally](https://askubuntu.com/a/1031424), integer `1` stands for `stdout` file descriptor, while `2` represents `stderr` file descriptor. 

```bash
echo Hello World > /dev/null
```

is same as

```bash
echo Hello World 1> /dev/null
```

**As for spacing**, integer is right next to redirection operator, but file can either be next to redirection operator or not, i.e., `command 2>/dev/null` or `command 2> /dev/null`.

The classical operator, `command > file` only redirects standard output, [several choices to redirect stderr](https://askubuntu.com/a/625230),

- Redirect stdout to one file and stderr to another file

```bash
command > out 2> error
```

- Redirect stdout to a file, and then redirect stderr to stdout **NO spacings in `2>&1`**

```bash
command > out 2>&1
```

- Redirect both to a file (not supported by all shells, but `bash` is OK)

```bash
command &> out
```

[In more technical terms](https://askubuntu.com/a/635069), `[integer]>&word` is called [Duplicating Output File Descriptor](https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_07_06)

We can [redirect output to a file and stdout simultaneously](https://stackoverflow.com/questions/418896/how-to-redirect-output-to-a-file-and-stdout)

```
program [arguments...] 2>&1 | tee outfile
```

## mv file with xargs

use `-I {}` to replace some str.

```bash
ls | grep 'config[0-9].txt' | xargs -I {} mv {} configs/
```

see more details in [mv files with | xargs](https://askubuntu.com/questions/487035/mv-files-with-xargs)

## google drive

refer to [Ubuntu 16.04 set up with google online account but no drive folder in nautilus](https://askubuntu.com/questions/838956/ubuntu-16-04-set-up-with-google-online-account-but-no-drive-folder-in-nautilus)

Note that you should run 

```bash
gnome-control-center online-accounts
```

in the command line, not to open the GUI.

## Bose Bluetooth

[Pair Bose QuietComfort 35 with Ubuntu over Bluetooth](https://askubuntu.com/questions/833322/pair-bose-quietcomfort-35-with-ubuntu-over-bluetooth)

## gvim fullscreen

refer to [Is there a way to turn gvim into fullscreen mode?](https://askubuntu.com/questions/2140/is-there-a-way-to-turn-gvim-into-fullscreen-mode)

In short, 

1. install wmctrl
2. map F11 via .vimrc

## thunderbird

1. [Special Gmail](Folders missing (Sent, Drafts, etc...) Only see Inbox and Trash. Please help.)
2. [Special Gmail (continued)](https://support.mozilla.org/zh-CN/kb/thunderbird-gmail)

## Ubuntu 16.04 create WiFi Hotpot

Refer to 

1. [3 Ways to Create Wifi Hotspot in Ubuntu 14.04 (Android Support)](http://ubuntuhandbook.org/index.php/2014/09/3-ways-create-wifi-hotspot-ubuntu/)
2. [How do I create a WiFi hotspot sharing wireless internet connection (single adapter)?](https://askubuntu.com/questions/318973/how-do-i-create-a-wifi-hotspot-sharing-wireless-internet-connection-single-adap)

几处不同：

1. 选择 `mode` 时，直接选择 `hotpot` 即可，后面也无需更改文件
2. 设置密码时位数不能少于 8 位
3. 连接 WiFi 时 似乎需要 enable wifi。

## `/dev/loopx`

refer to [What is /dev/loopx?(https://askubuntu.com/questions/906581/what-is-dev-loopx).

## 惊魂扩容

一直想扩容来着，但总是下不了决心。今天决定了，参考 google 搜索“Ubuntu 扩容”的前几条结果，便开始干了。

1. 采用启动 U 盘，因为根目录在使用状态，幸好启动 U 盘还在。
2. 使用 Gparted 时有个大大的 warning，说对含 /boot 分区的硬盘进行操作可能会不能正常启动，有点吓到了，最后还是狠下心继续下去了。
3. 网上有人说，不要用 Gparted 对 Windows 进行压缩，而应该在 Windows 中进行压缩，可是此时已经开始了，想中断但怕造成更严重的后果，幸好最后启动 Windows 时只是多了步检查硬盘，并没有不能启动的状况。

中间提心吊胆，好在最后顺利扩容完成。

## 移动硬盘重命名

通过

```bash
gnome-disks
```

进行设置，详见[How to change hard drive name](https://askubuntu.com/questions/904561/how-to-change-hard-drive-name/904564)

## remove broken link

```bash
find -L . -name . -o -type d -prune -o -type l -exec rm {} + 
```

[Delete all broken symbolic links with a line?](https://stackoverflow.com/questions/22097130/delete-all-broken-symbolic-links-with-a-line)

## wget

### wget a series of files in order

下载连续编号的文件，如

```
wget http://work.caltech.edu/slides/slides{01..18}.pdf
```

参考 [Wget a series of files in order](https://askubuntu.com/questions/240702/wget-a-series-of-files-in-order)

### `wget` vs `curl`

`wget` 不用添加 `-O` 就可以将下载的文件存储下来，但是 `curl` 并不默认将下载的文件存入本地文件，除非加上 `-o` 选项，而 `wget` 的 `-O` 只是为了更改文件名。

比如[这里](https://github.com/huan/docker-wine/blob/54e7ba2f042a59de72a06bafc37f1fb8c554541e/Dockerfile#L36)，直接将下载的内容输出到下一个命令

```bash
curl -sL https://dl.winehq.org/wine-builds/winehq.key | apt-key add - 
```

更多比较详见 [What is the difference between curl and wget?](https://unix.stackexchange.com/questions/47434/what-is-the-difference-between-curl-and-wget)



## hydrogen specify the conda envirnoment

just need to run

```
source activate thisenv
python -m ipykernel install --user --name thisenv
```

and only once, hydrogen will remember this!!

ref to [How to specify the conda environment in which hydrogen (jupyter) starts?](https://github.com/nteract/hydrogen/issues/899)

## show long character usernames which consists of `+`

refer to [ps aux for long charactered usernames shows a plus sign](https://askubuntu.com/questions/523673/ps-aux-for-long-charactered-usernames-shows-a-plus-sign)


```
ps axo user:20,pid,pcpu,pmem,vsz,rss,tty,stat,start,time,comm
alias psaux='ps axo user:20,pid,pcpu,pmem,vsz,rss,tty,stat,start,time,comm'
```

## remove the first character

```bash
${string:1}
```

```bash
list=""
for nc in {2..10}; do
  for nf in 5 10 15; do
    list="$list,acc-$nc-$nf"
    #https://stackoverflow.com/questions/6594085/remove-first-character-of-a-string-in-bash
    echo ${list:1}
  done
done
```

refer to [Remove first character of a string in Bash](https://stackoverflow.com/questions/6594085/remove-first-character-of-a-string-in-bash)

## convert imgs to pdf

```bash
ls -1 ./*jpg | xargs -L1 -I {} img2pdf {} -o {}.pdf
pdftk likelihoodfree-design-a-discussion-{1..13}-1024.jpg.pdf cat output likelihoodfree-design-a-discussion.pdf
```

注意这里需要用 `ls -1`，如果 `ll` 则第一行会有 `total xxx` 的信息，即 `ll | wc -l` 等于 `ls -1 | wc -l` + 1，而且在我的 Ubuntu 18.04 中，`ll` 甚至还会列出 

```bash
./
../
```

这一点在服务器上没看到。


## modify pdf metadata via `pdftk`

```bash
pdftk input.pdf dump_data output metadata
# edit metadata
pdftk input.pdf update_info metadata output output.pdf
```


## zip 文件解压乱码

别人在 Windows 下加压的文件，通过微信发送，在 Ubuntu 16.04 中解压时文件名乱码。

采用 `unar your.zip`

参考 [Linux文件乱码](https://www.findhao.net/easycoding/1605)




## 图片处理 

### 拼接

水平方向

```bash
convert +append *.png out.png 
```

垂直方向

```bash
convert -append *.png out.png
```

参考 [How do I join two images in Ubuntu?](https://askubuntu.com/a/889772)

## 文本文件查看

`cut`: select by columns

参考 [10 command-line tools for data analysis in Linux](https://opensource.com/article/17/2/command-line-tools-data-analysis-linux)

## 文本文件拼接

### 按列

```bash
paste file1 file2 > outputfile
```

### 按行

```bash
cat file1 file2 > outputfile
```

## 视频处理

### 去除音频

参考 [如何使用ffmpeg去除视频声音？](https://hefang.link/article/how-remove-voice-with-ffmpeg.html)

```bash
ffmpeg -i .\input.mp4 -map 0:0 -vcodec copy out.mp4
```

### 慢速播放和快速播放

参考 [ffmpeg 视频倍速播放 和 慢速播放](https://blog.csdn.net/ternence_hsu/article/details/85865718)

## fuseblk

发现使用 onedrive 同步文件时，有时候并不能够同步。猜测可能是因为文件太小，比如文件夹 `test` 中仅有 `test.md` 文件（仅70B），而此时查看 `test` 大小，竟然为 0 B，因为根据常识，一般文件夹都是 4.0k，或者有时 8.0k 等等，具体原因参考 [Why does every directory have a size 4096 bytes (4 K)?](https://askubuntu.com/questions/186813/why-does-every-directory-have-a-size-4096-bytes-4-k)

但我现在问题是文件夹竟然是 0B，猜测这是无法同步的原因。

后来在上述问题的回答的评论中 @Ruslan 提到

> @phyloflash some filesystems (e.g. NTFS) store small files in the file entries themselves (for NTFS it's in the MFT entry). This way their contents occupy zero allocation blocks, and internal fragmentation is reduced. – Ruslan Nov 2 at 9:03

猜测这是文件系统的原因，因为此时文件夹刚好位于移动硬盘中，所以可能刚好发生了所谓的 “internal fragmentation is reduced”。

于是准备查看移动硬盘的 file system 来验证我的想法，这可以通过 `df -Th` 实现，具体参考 [7 Ways to Determine the File System Type in Linux (Ext2, Ext3 or Ext4)](https://www.tecmint.com/find-linux-filesystem-type/)

然后竟然发现并不是期望中的 NTFS，而是 fuseblk，[東海陳光劍的博客](http://blog.sina.com.cn/s/blog_7d553bb501012z3l.html)中解释道

> fuse是一个用户空间实现的文件系统。内核不认识。fuseblk应该就是使用fuse的block设备吧，系统中临时的非超级用户的设备挂载好像用的就是这个。

最后发现，onedrive 无法同步的原因可能并不是因为 0 byte 的文件夹，而是因为下面的命名规范，虽然不是需要同步的文件，而是之前很久的文件，但可能onedrive就在之前这个不规范命名的文件上崩溃了。

## windows 命名规范

在使用 [onedrive](https://github.com/skilion/onedrive) 同步时，一直会出现碰到某个文件崩溃。查了一下才知道是需要遵循 [Windows 命名规范](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file?redirectedfrom=MSDN)，其中有两条很重要

- Do not assume case sensitivity. For example, consider the names OSCAR, Oscar, and oscar to be the same, even though some file systems (such as a POSIX-compliant file system) may consider them as different. Note that NTFS supports POSIX semantics for case sensitivity but this is not the default behavior. 
- The following reserved characters:
  - < (less than)
  - > (greater than)
  - : (colon)
  - " (double quote)
  - / (forward slash)
  - \ (backslash)
  - | (vertical bar or pipe)
  - ? (question mark)
  - * (asterisk)

## 后台运行

- `jobs -l` 返回后台运行程序的 `PID`，refer to [How to get PID of background process?](https://stackoverflow.com/questions/1908610/how-to-get-pid-of-background-process) 

但是 `jobs` [只显示属于当前 shell 的后台程序](https://superuser.com/a/607219), 如果重新登录，则不会显示后台程序，详见 [`jobs` command doesn't show any background processes](https://superuser.com/questions/607218/jobs-command-doesnt-show-any-background-processes)

## `htop`

A much more powerful command than `top`, refer to [Find out what processes are running in the background on Linux](https://www.cyberciti.biz/faq/find-out-what-processes-are-running-in-the-background-on-linux/)

## different CUDA version shown by nvcc and NVIDIA-smi

refer to [Different CUDA versions shown by nvcc and NVIDIA-smi](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi)

> CUDA has 2 primary APIs, the runtime and the driver API. Both have a corresponding version
>
> - The necessary support for the driver API (e.g. libcuda.so on linux) is installed by the GPU driver installer.
> - The necessary support for the runtime API (e.g. libcudart.so on linux, and also nvcc) is installed by the CUDA toolkit installer (which may also have a GPU driver installer bundled in it).

`nvidia-smi`: installed by the GPU driver installer, and generally has the GPU driver in view, not anything installed by the CUDA toolkit installer.
`nvcc`: the CUDA compiler-driver tool that is installed with the CUDA toolkit, will always report the CUDA runtime version that it was built to recognize. 

## 共享打印机

现有台 HP-Deskjet-1050-J410-series 打印机，通过 USB 接口。直接连接在 Ubuntu 上是可以实现打印功能的，现在想贡献给局域网内的其他设备，参考 [使用Linux共享打印机](https://www.jianshu.com/p/a1c4fc6d9ce8)，主要步骤为

1. 安装 CUPS 服务，`sudo apt-get install cups` 并启动，`sudo service cups start`
2. 在 `127.0.0.1:631` 的 `Administration >> Advanced` 勾选 `Allow printing from the Internet`，并保存。
3. 打开防火墙，`sudo ufw allow 631/tcp`

在同一局域网内的 Windows 设备中，添加该打印机，地址即为Ubuntu中浏览器的地址，注意将 `127.0.0.1` 换成局域网 ip。如果顺利的话，添加后需要添加驱动程序，可以在 HP 官网下载。

## Install Fira

[Fira for Metropolis theme](https://github.com/matze/mtheme/issues/280)

[Install the Fira Font in Ubuntu](https://stevescott.ca/2016-10-20-installing-the-fira-font-in-ubuntu.html)

and some introduction: [Fira Code —— 专为编程而生的字体](https://zhuanlan.zhihu.com/p/65362086)

## Proxy for Gmail in Thunderbird

Setting a proxy for the thunderbird is quite straigtforward, but not all mail accounts need the proxy, only gmail in my case. I am considering if it is possible to set up a proxy for gmail separately. Then I found that setting proxy by PAC file might work inspired by [Gmail imap/smtp domains to connect via proxy](https://support.google.com/mail/forum/AAAAK7un8RUCGQj5uPgJoo), since PAC file can customize the visited url. 

Then I need to learn [how to write a PAC file](https://findproxyforurl.com/example-pac-file/), although later I directly export the rules written in SwitchyOmega to a PAC file.

Once PAC is done, I need to write its location url, seems impossible to directly write a local path. One easy way is to open port 80 to access my laptop, which maybe need apache or nginx, but both of them are overqualified. A simple way is

```bash
sudo python -m SimpleHTTPServer 80
```

found in[Open port 80 on Ubuntu server](https://askubuntu.com/questions/646293/open-port-80-on-ubuntu-server)


## proxy for apt

`proxychains` seems not work well before `sudo` or after `sudo`, and I dont want to add a system proxy permanently, then I found a temporary way,

```bash
sudo http_proxy='http://user:pass@proxy.example.com:8080/' apt-get install package-name
```

refer to [how to install packages with apt-get on a system connected via proxy?](https://askubuntu.com/questions/89437/how-to-install-packages-with-apt-get-on-a-system-connected-via-proxy)

## 上次重启时间

```bash
last reboot
# or
uptime --since # actually the first line of `top`
```

参考 [How long has my Linux system been running?](https://unix.stackexchange.com/questions/131775/how-long-has-my-linux-system-been-running)

## number of cores

> CPUs = Threads per core X cores per socket X sockets

quick way to check

```bash
nproc --all
```

more details

```bash
lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('
```

refer to [How to know number of cores of a system in Linux?](https://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux)

## 顶部栏系统监测信息

通过 gnome-shell extension: [gnome-shell-system-monitor-applet](https://github.com/paradoxxxzero/gnome-shell-system-monitor-applet) 实现

不过目前有个小问题，字体略小，尝试通过 gnome-tweaks 中的 scaling 来改变字体大小，但似乎对这些字体仍不适用，先将就用着吧。
