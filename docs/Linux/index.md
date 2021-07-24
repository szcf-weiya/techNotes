# Linux Notes

## System/Hardware Info

- check linux distribution (refer to [How To Find Out My Linux Distribution Name and Version](https://www.cyberciti.biz/faq/find-linux-distribution-name-version-number/))
  	- `cat /etc/os-release` or `cat /ect/*-release`
  	- `lsb_release -a`
  	- `hostnamectl`

!!! note "Linux Distributions"

    根据包管理器进行的分类，主流的发行版有

    - apt: Debian, Ubuntu, Linux Mint
    - yum: CentOS, Fedora
    - YaST: openSUSE
    - Pacman: Manjaro、ArchLinux

    另外，根据这道 [Shell 练习题](../shell/#logical-operation)，

    - Redhat Series: Fedora, Gentoo, Redhat
    - Suse Series: Suse, OpenSuse
    - Debian Series: Ubuntu, Mint, Debian

    参考 

    - [Major Distributions: An overview of major Linux distributions and FreeBSD](https://distrowatch.com/dwres.php?resource=major)
    - [知乎：Linux 各大发行版有什么特色？](https://www.zhihu.com/question/24261540)

    目前只接触 Ubuntu（个人笔记本） 和 CentOS, Rocky（服务器），所以本笔记主要针对这两种。

- check kernel version:
	- `uname -a` or `uname -mrs`
	- `cat /proc/version`
- how long has the system been running (refer to [How long has my Linux system been running?](https://unix.stackexchange.com/questions/131775/how-long-has-my-linux-system-been-running)):
	- `last reboot`
	- `uptime --since`, which is actually the first line of `top`
- obtain [MAC address :link:](https://help.ubuntu.com/stable/ubuntu-help/net-macaddress.html): ` ifconfig | grep ether`, note that different MAC addresses for WIFI and Ethernet.
- check number of cores (refer to [How to know number of cores of a system in Linux?](https://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux)):
    - quick way: `nproc --all`
    - more details: `lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('`

!!! note "CPU vs Thread vs Core vs Socket"
    - CPU: Central Processing Unit。概念比较宽泛，不同语境有不同含义，如 `lscpu` 便指 thread 个数。`CPUs` = `Threads per core` * `cores per socket` * `sockets`
    - CPU Socket: CPU 是通过一个插槽安装在主板上的，这个插槽就是 Socket;
    - Core: 一个 CPU 中可以有多个 core，各个 core 之间相互独立，且可以执行并行逻辑，每个 core 都有单独的寄存器，L1, L2 缓存等物理硬件。
    - Thread: 并不是物理概念，而是软件概念，本质上是利用 CPU 空闲时间来执行其他代码，所以其只能算是并发，而不是并行。
    - vCPU: 常见于虚拟核，也就是 Thread
    
    ```bash
    G40 $ lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('
    CPU(s):                          4
    Thread(s) per core:              2
    Core(s) per socket:              2
    Socket(s):                       1
    ```
    表明其为 2 核 4 线程。

    ```bash
    T460P $ $ lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('
    CPU(s):              4
    Thread(s) per core:  1
    Core(s) per socket:  4
    Socket(s):           1
    ```
    表明其为 4 核 4 线程。

    参考 [三分钟速览cpu,socket,core,thread等术语之间的关系](https://cloud.tencent.com/developer/article/1736628)

- check last logged users: `last`, but the user field only shows 8 characters. To check the full name, use `last -w` instead. Refer to [last loggedin users in linux showing 8 characters only - Server Fault](https://serverfault.com/questions/343740/last-loggedin-users-in-linux-showing-8-characters-only)

## Add User

```bash
$ useradd -m -s /bin/bash userName
$ passwd userName
```

Or explicitly specify the password with

```bash
useradd -p $(openssl passwd -1 "PASSWORD") -m userName
```

where `-1` means to use the MD5 based BSD password algorithm 1, see `man openssl-passwd` for more details.

Create users in batch mode,

```bash
for i in {01..14}; do useradd -p $(openssl passwd -1 "PASSWORD\$") -m "project$i"; done
```

where symbol `$` (if any) needs to be escaped by `\`.

!!! warning
    As `man useradd` notes,
    > `-p` option is not recommended because the password (or encrypted password) will be visible by users listing the processes.

增加 sudo 权限

```bash
$ sudoedit /etc/sudoers
```

```diff
# Allow members of group sudo to execute any command
%sudo	ALL=(ALL:ALL) ALL
+weiya ALL=(ALL) NOPASSWD:ALL
+szcf715 ALL=(ALL) ALL
```

其中 `NOPASSWD` 表示用户 `weiya` 在使用 `sudo` 时无需输入密码，而 `szcf715` 则需要输入密码才能使用 `sudo`.

`man sudoers` 给了一些具体的设置例子，搜索 `example sudoers`.

参考 [https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04](http://blog.csdn.net/linuxdriverdeveloper/article/details/7427672)

## Locale

> 区域设置（locale），也称作“本地化策略集”、“本地环境”，是表达程序用户地区方面的软件设定。不同系统、平台、与软件有不同的区域设置处理方式和不同的设置范围，但是一般区域设置最少也会包括语言和地区。区域设置的内容包括：数据格式、货币金额格式、小数点符号、千分位符号、度量衡单位、通货符号、日期写法、日历类型、文字排序、姓名格式、地址等等。
> source: [维基百科](https://zh.wikipedia.org/wiki/%E5%8C%BA%E5%9F%9F%E8%AE%BE%E7%BD%AE)

locale 生效的顺序为

1. `LANGUAGE`：指定个人对语言环境值的主次偏好，在 Ubuntu 中有这个环境变量，但似乎在 CentOS7.4 服务器上没有这个变量
2. `LC_ALL`: 这不是一个环境变量，是一个可被C语言库函数setlocale设置的宏，其值可覆盖所有其他的locale设定。因此缺省时此值为空
3. `LC_xxx`: 可设定locale各方面（category）的值，可以覆盖 `LANG` 的值。
4. `LANG`: 指定默认使用的locale值

如若设置不当，可能会出现

```bash
$ locale
locale: Cannot set LC_CTYPE to default locale: No such file or directory
locale: Cannot set LC_MESSAGES to default locale: No such file or directory
locale: Cannot set LC_ALL to default locale: No such file or directory
LANG=C.UTF-8
LC_CTYPE=C.UTF-8
LC_NUMERIC=en_US.UTF-8
LC_TIME=en_US.UTF-8
LC_COLLATE="C.UTF-8"
LC_MONETARY=en_US.UTF-8
LC_MESSAGES="C.UTF-8"
LC_PAPER=en_US.UTF-8
LC_NAME=en_US.UTF-8
LC_ADDRESS=en_US.UTF-8
LC_TELEPHONE=en_US.UTF-8
LC_MEASUREMENT=en_US.UTF-8
LC_IDENTIFICATION=en_US.UTF-8
LC_ALL=
```

则可以通过

```bash
export LC_ALL=en_US.UTF-8
```

来解决这个问题，这个可以写进 `.bashrc` 文件中，并且不需要 sudo 权限，而 [How do I fix my locale issue?](https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue) 中提到的几种方法需要 sudo 权限。

## GNOME 

GNOME (originally an acronym for GNU Network Object Model Environment) is a desktop environment for Unix-like operating systems. [:material-wikipedia:](https://en.wikipedia.org/wiki/GNOME)

The version on my T460p is 3.28.2, which can be seen from About.

### GNOME Shell

GNOME Shell is the graphical shell of the GNOME desktop environment. It provides basic functions like launching applications, switching between windows and is also a widget engine. [:material-wikipedia:](https://en.wikipedia.org/wiki/GNOME_Shell). User interface elements provided by GNOME Shell include the Panel at the top of the screen, the Activities Overview, and Message Tray at the bottom of the screen. [:link:](https://extensions.gnome.org/about/)

The version on my T460p is 

```bash
$ gnome-shell --version
GNOME Shell 3.28.4
```

### GNOME Shell Extensions

[GNOME Shell Extensions](https://extensions.gnome.org/about/) are small pieces of code written by third party developers that modify the way GNOME works. They are similar to Chrome Extensions or Firefox Addons. We can install the extensions via [extensions.gnome.org](https://extensions.gnome.org/) in Firefox. After installation, we can disable or enable, or even configure on such website, alternatively, we can use `gnome-tweaks` to control them.

### Lunar Date

Here is a plugin to show Chinese Lunar Date: [Lunar Calendar 农历](https://extensions.gnome.org/extension/675/lunar-calendar/). Since here are some latest comments, I guess it would be OK.

However, the first installation attempt failed, it shows `ERROR`. Then I realized that I might need to install the dependency mentioned in the plugin page,

```bash
sudo apt install gir1.2-lunar-date-2.0
```

Then reinstall the plugin, it succeed! But interestingly, the Chinese characters are shown as Pinyin (see the following left image)

Before | After
-- | --
![Screenshot from 2021-05-03 14-04-34](https://user-images.githubusercontent.com/13688320/116846751-a65ae000-ac1b-11eb-9c40-31ba384f63db.png)|![Screenshot from 2021-05-03 14-40-47](https://user-images.githubusercontent.com/13688320/116847724-b2e03800-ac1d-11eb-9700-bccb1a4e25f2.png)

Then I found the same issue in [an older post](https://forum.ubuntu.org.cn/viewtopic.php?t=308968)

A solution is 

```bash
@GuanGGuanG
copy
/usr/share/locale/zh_CN/LC_MESSAGES/liblunar.mo
to
/usr/share/locale/en/LC_MESSAGES/
就可以在英文环境下正常显示了
```

Although no found `liblunar.mo`, there is 

```bash
$ pwd
/usr/share/locale/zh_CN/LC_MESSAGES
$ ll | grep lunar
-rw-r--r-- 1 root root   4746 Nov 12  2016 lunar-date.mo
```

then

```bash
$ sudo cp lunar-date.mo ../../en/LC_MESSAGES/
```

It cannot take effects immediately, the natural way is to reboot. But currently I do not want to reboot, and then I tried to reinstall the plugin in Firefox, but not work.

Then I tried to reload locale since the modification seems related to locale, so I found [this answer](https://unix.stackexchange.com/questions/108514/reload-etc-default-locale-without-reboot) and tried

```bash
$ . /etc/default/locale
```

but not work.

Then I realized that it might be necessary to reload GNOME Shell, so I found [How to restart GNOME Shell from command line?](https://askubuntu.com/questions/100226/how-to-restart-gnome-shell-from-command-line), and tried

```bash
$ gnome-shell --replace &
```

It works, as shown in the above right figure. A minor side change is that the English colon in the time `14:37` seems to change to the Chinese colon.

### System Monitor

通过 gnome-shell extension: [gnome-shell-system-monitor-applet](https://github.com/paradoxxxzero/gnome-shell-system-monitor-applet) 实现

不过目前有个小问题，字体略小，尝试通过 gnome-tweaks 中的 scaling 来改变字体大小，但似乎对这些字体仍不适用，先将就用着吧。

### unblack lock screen

按 `Win+L` 锁屏后，很快就直接变黑了。因为感觉屏保还挺好看的，所以并不想直接黑屏。参考 [GNOME3锁屏和锁屏后，如何设置屏幕常亮？ - Eglinux的回答 - 知乎](https://www.zhihu.com/question/276118015/answer/656464977)，安装 [Unblank lock screen.](https://extensions.gnome.org/extension/1414/unblank/)

更简单的技巧是长按 `Win+L`，似乎确实不会直接黑屏，然后会直接采用设置的关屏时间（Setting > Power），参考 [GNOME3锁屏和锁屏后，如何设置屏幕常亮？ - dale的回答 - 知乎](https://www.zhihu.com/question/276118015/answer/496472138)。

## install win on ubuntu

参考[http://www.linuxdeveloper.space/install-windows-after-linux/](http://www.linuxdeveloper.space/install-windows-after-linux/)


## unable to resolve host

参考[http://blog.csdn.net/ichuzhen/article/details/8241847](http://blog.csdn.net/ichuzhen/article/details/8241847)

## 初始化服务器

1. 新建用户，sudo
2. 添加sources.list,gpg
3. 安装R
4. 安装Rstudioserver（成功！！！哎。。搞了一下午就是因为上午莫名其妙更新了Ubuntu，不要手贱！！）

## shared objects `.so` (dynamic library)

As said in [Where do executables look for shared objects at runtime?](https://unix.stackexchange.com/questions/22926/where-do-executables-look-for-shared-objects-at-runtime), when it's looking for a dynamic library (`.so` file) the linker tries

- directories listed in the `LD_LIBRARY_PATH`
- directories listed in the executable's rpath, such as via `$ORIGIN/../lib`
- directories on the system search path, which consists of the entries in `/etc/ld.so.conf` plus `/lib` and `/usr/lib`

Then there are several ways to fix the NotFound error,

```bash
# method 1
sudo ln -s /where/your/lib/*.so /usr/lib
sudo ldconfig
# method 2
export LD_LIBRARY_PATH=/where/your/lib:$LD_LIBRARY_PATH`
sudo ldconfig
# method 3
sudo echo "where/your/lib" >> /etc/ld.so.conf
sudo ldconfig
```

## could not get lock /var/lib/dpkg/lock -open

```
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

如果不行，重启。


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


## Terminator

- hostname 的颜色, 去掉 `.bashrc` 中

```bash
##force_color_prompt=yes
```

的注释

- hide hostname, `weiya@weiya-ThinkPad-T460p:`

edit the following line in the `.bashrc` as follows

```bash
if [ "$color_prompt" = yes ]; then
    #PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
    PS1='\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
```

before hide and after hide 

![image](https://user-images.githubusercontent.com/13688320/123048663-37d31b00-d431-11eb-8ebf-afb97f758191.png)

- 颜色背景色等，直接右键设置，右键设置完成之后便有了一个配置文件，`~/.config/terminator/config`.

## Linux 杀进程

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
- `CMD`: see args.  (alias args, command). when the arguments to that command cannot be located, 会被中括号 `[]` 包起来


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

## 文件权限

采用`ls -l` 便可以查看文件(夹)权限，比如

```bash
-rw-rw-r--  1 weiya weiya    137969 3月   8  2017 font.txt
-rw-r--r--  1 root  root      35792 12月 26 23:50 geckodriver.log
-rw-r--r--  1 root  root     327350 12月 27 01:38 ghostdriver.log
```
7列的含义分别是（参考[http://blog.csdn.net/jenminzhang/article/details/9816853](http://blog.csdn.net/jenminzhang/article/details/9816853)）

1. 文件类型和文件权限
  - 文件类型由第一个字母表示，常见的有 `d`(目录)，`-`(文件)，`l`(链接)
  - 权限分为三段，每三个字符一段，分别表示，文件所有者 `u`、文件所属组 `g`、其他用户 `o`对该文件的权限，其中
    - `r`: 可读，等于 4
    - `w`: 可写，等于 2
    - `x`: 可执行，等于 1
    - `-`: 无权限，等于 0
    - `s`: set user or group ID on execution (s)
    - `X`: execute/search only if the file  is a directory or already has  execute permission for some user
    - `t`: restricted deletion flag or sticky bit
2. 文件链接个数
3. 文件所有者
4. 文件所在群组
5. 文件长度
6. 时间
7. 文件名称


采用chmod修改权限（参考[http://www.linuxidc.com/Linux/2015-03/114695.htm](http://www.linuxidc.com/Linux/2015-03/114695.htm)），如

```bash
chmod -R 700 Document/
chmod -R [ugoa...][[+-=][perms...]] # refer to `man chmod` for more details
```

其中 `-R` 表示递归，`perms` 为上述 `rwxXst`，而 `a` 表示所有用户，即 `ugo`.

采用 chown 改变所有者，比如

```bash
chown -R username:users Document/
```

`chmod g+s .` 会使得当前文件夹 `.` 中所有新建文件或文件夹都继承 `.` 的 group，而不是创建者所属的 group，所以这一般配合 `chgrp` 使用。参考 ['chmod g+s' command](https://unix.stackexchange.com/questions/182212/chmod-gs-command)

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

## `user` vs. `sys`

- `time` commands return three times, named `real`, `user` and `sys`, the detailed explanation refers to [What do 'real', 'user' and 'sys' mean in the output of time(1)?](https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1)
- `user space` vs `kernel space`: [维基百科](https://zh.wikipedia.org/wiki/%E4%BD%BF%E7%94%A8%E8%80%85%E7%A9%BA%E9%96%93)说，在操作系统中，虚拟内存通常会被分成用户空间（英语：User space，又译为使用者空间），与核心空间（英语：Kernel space，又译为内核空间）这两个区段。

## control android phone by PC's mouse and keyboard

[How to Control Your Android Using Your Computer’s Mouse and Keyboard](https://www.makeuseof.com/tag/control-android-using-computers-mouse-keyboard/)


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

## Font

### `fc-list`

view installed fonts

```bash
# only print the font-family
$ fc-list : family
# add language selector
$ fc-list : family lang=zh
...
Fira Sans,Fira Sans UltraLight
Fira Sans,Fira Sans Light
Noto Serif CJK KR,Noto Serif CJK KR ExtraLight
# with format option, get the family names of all the fonts (note that the above family also specify the detailed style)
$ fc-list --format='%{family[0]}\n' :lang=zh | sort | uniq
...
文泉驿等宽微米黑
文泉驿等宽正黑
新宋体
```

refer to [fc-list command in Linux with examples](https://www.geeksforgeeks.org/fc-list-command-in-linux-with-examples/)

### Install Local Fonts

以安装仿宋和黑体为例，这是[本科毕业论文模板](https://hohoweiya.xyz/zju-thesis/src/zju-thesis.pdf)所需要的字体，字体文件已打包

```bash
$ wget -c https://sourceforge.net/projects/zjuthesis/files/fonts.tar.gz/download -O fonts.tar.gz
$ tar xvzf fonts.tar.gz
fonts/STFANGSO.TTF
fonts/
fonts/simhei.ttf
$ sudo mkdir -p /usr/share/fonts/truetype/custom/
$ sudo mv fonts/* /usr/share/fonts/truetype/custom/
$ sudo fc-cache -f -v
```

安装完成后，

```bash
$ fc-list :lang=zh
/usr/share/fonts/truetype/custom/simhei.ttf: SimHei,黑体:style=Regular,Normal,obyčejné,Standard,Κανονικά,Normaali,Normál,Normale,Standaard,Normalny,Обычный,Normálne,Navadno,Arrunta
/usr/share/fonts/truetype/custom/STFANGSO.TTF: STFangsong,华文仿宋:style=Regular
```

### Some Free Fonts

- [Mozilla's Fira Type Family](https://github.com/mozilla/Fira)
    - [Fira for Metropolis theme](https://github.com/matze/mtheme/issues/280)
    - [Fira Code](https://github.com/tonsky/FiraCode)
        - [知乎：Fira Code —— 专为编程而生的字体](https://zhuanlan.zhihu.com/p/65362086)

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


## Unable to lock the administration directory (/var/lib/dpkg/) is another process using it?

[Unable to lock the administration directory (/var/lib/dpkg/) is another process using it?](https://askubuntu.com/questions/15433/unable-to-lock-the-administration-directory-var-lib-dpkg-is-another-process)

## mv file with xargs

use `-I {}` to replace some str.

```bash
ls | grep 'config[0-9].txt' | xargs -I {} mv {} configs/
```

see more details in [mv files with | xargs](https://askubuntu.com/questions/487035/mv-files-with-xargs)

see also: [xargs命令_Linux xargs 命令用法详解：给其他命令传递参数的一个过滤器](http://man.linuxde.net/xargs)


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



## Ubuntu 16.04 create WiFi Hotpot

Refer to

1. [3 Ways to Create Wifi Hotspot in Ubuntu 14.04 (Android Support)](http://ubuntuhandbook.org/index.php/2014/09/3-ways-create-wifi-hotspot-ubuntu/)
2. [How do I create a WiFi hotspot sharing wireless internet connection (single adapter)?](https://askubuntu.com/questions/318973/how-do-i-create-a-wifi-hotspot-sharing-wireless-internet-connection-single-adap)

几处不同：

1. 选择 `mode` 时，直接选择 `hotpot` 即可，后面也无需更改文件
2. 设置密码时位数不能少于 8 位
3. 连接 WiFi 时 似乎需要 enable wifi。

## `/dev/loopx`

refer to [What is /dev/loopx?](https://askubuntu.com/questions/906581/what-is-dev-loopx).

## 惊魂扩容

一直想扩容来着，但总是下不了决心。今天决定了，参考 google 搜索“Ubuntu 扩容”的前几条结果，便开始干了。

1. 采用启动 U 盘，因为根目录在使用状态，幸好启动 U 盘还在。
2. 使用 Gparted 时有个大大的 warning，说对含 /boot 分区的硬盘进行操作可能会不能正常启动，有点吓到了，最后还是狠下心继续下去了。
3. 网上有人说，不要用 Gparted 对 Windows 进行压缩，而应该在 Windows 中进行压缩，可是此时已经开始了，想中断但怕造成更严重的后果，幸好最后启动 Windows 时只是多了步检查硬盘，并没有不能启动的状况。

中间提心吊胆，好在最后顺利扩容完成。

## 移动硬盘重命名

终端输入

```bash
gnome-disks
```

在设置齿轮图标中选择 `Edit Mount Options`，修改 `Mount Point`。注意重新挂载后才能生效。

详见[How to change hard drive name](https://askubuntu.com/questions/904561/how-to-change-hard-drive-name/904564)

## remove broken link

```bash
find -L . -name . -o -type d -prune -o -type l -exec rm {} +
```

[Delete all broken symbolic links with a line?](https://stackoverflow.com/questions/22097130/delete-all-broken-symbolic-links-with-a-line)

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

## modify pdf metadata via `pdftk`

```bash
pdftk input.pdf dump_data output metadata
# edit metadata
pdftk input.pdf update_info metadata output output.pdf
```

## 文本文件查看

`cut`: select by columns

参考 [10 command-line tools for data analysis in Linux](https://opensource.com/article/17/2/command-line-tools-data-analysis-linux)


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

## 后台运行

- `jobs -l` 返回后台运行程序的 `PID`，refer to [How to get PID of background process?](https://stackoverflow.com/questions/1908610/how-to-get-pid-of-background-process)

但是 `jobs` [只显示属于当前 shell 的后台程序](https://superuser.com/a/607219), 如果重新登录，则不会显示后台程序，详见 [`jobs` command doesn't show any background processes](https://superuser.com/questions/607218/jobs-command-doesnt-show-any-background-processes)



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

## proxy for apt

`proxychains` seems not work well before `sudo` or after `sudo`, and I dont want to add a system proxy permanently, then I found a temporary way,

```bash
sudo http_proxy='http://user:pass@proxy.example.com:8080/' apt-get install package-name
```

refer to [how to install packages with apt-get on a system connected via proxy?](https://askubuntu.com/questions/89437/how-to-install-packages-with-apt-get-on-a-system-connected-via-proxy)



## .netrc

为了学习 RL，在听了周博磊在 B 站的视频后，准备玩下[示例代码](https://github.com/cuhkrlcourse/RLexample)，但是在终端中创建新 conda 环境时，

```bash
conda create --name RL python=3
```

总是报错，

> Collecting package metadata (current_repodata.json): failed
>
> ProxyError: Conda cannot proceed due to an error in your proxy configuration.
> Check for typos and other configuration errors in any '.netrc' file in your home directory,
> any environment variables ending in '_PROXY', and any other system-wide proxy
> configuration settings.

其中提到一个 `.netrc`，没想到自己竟然还真的有这个文件，看了下内容，只有两条，

```bash
machine api.heroku.com
...
machine git.heroku.com
...
```

这才意识到很早无意识中折腾 heroku 时创建的。那这个文件是干嘛的呢，[查了一下发现](https://stackoverflow.com/questions/21828495/purpose-of-the-netrc-file)

> This is a file that is often used by Unix programs to hold access details for remote sites. It was originally created for use with FTP.

最后这个问题是直接把 .bashrc 中所有的代理去掉了.

## possible errors using `apt-get`

[How do I resolve unmet dependencies after adding a PPA?](https://askubuntu.com/questions/140246/how-do-i-resolve-unmet-dependencies-after-adding-a-ppa)

## 删除 Hotspot

升级到 Ubuntu 18.04 后，开机自动连接到 Hotspot，每次需要手动禁止并改成 Wifi 连接，这个可以直接删除保存好的 Hotspot 连接

```bash
cd /etc/NetworkManager/system-connections/
sudo rm Hotspot
```

参考 [How to remove access point from saved list](https://askubuntu.com/questions/120415/how-to-remove-access-point-from-saved-list/120447)

## 移动 SSD 硬盘

因为硬盘太小，而移动硬盘读写文件速度实在有点慢，然后看到有[移动 SSD 硬盘](https://zhuanlan.zhihu.com/p/61083491)，于是便入手了一个。

当然首先确定了，自己笔记本有 USB3.0 接口，虽然不是 USB3.1。（不过好像 USB3.0 也叫作 USB3.1 gen1，而真正的 USB3.1 叫做 USB3.1 gen2），这个可以通过

```bash
$ lsusb -t
```

来看[接口的情况](https://superuser.com/questions/781398/how-can-i-be-sure-that-ive-plugged-a-device-into-a-usb-3-port)，当然也直接搜了电脑型号来看具体配置、

货到手后，一开始插上时，说

> Mount error: unknown filesystem type ‘exfat’

本来以为需要什么格式转化之类的，后来发现[解决方案](https://better-coding.com/solved-mount-error-unknown-filesystem-type-exfat/#:~:text=Cause%20Some%20SD%20Cards%20and,%2Dfuse%20and%20exfat%2Dutils.)挺简单的，

```bash
sudo apt-get install exfat-fuse exfat-utils
```

但是后来发现这个格式很多地方会出现不兼容，比如

1. 解压某个文件时，报出 `Cannot set modif./access times`，而这个在正常磁盘以及已有的移动硬盘中都能正常解压
2. 不能创建带有 `:` 的文件夹，这应该是遵循 Windows 的标准，但是 Linux 命名标准没有遵循 Windows，所以造成有些文件复制不过去。

最后决定格式化为 Linux 磁盘的格式，这个其实也挺简单的，进入 `gnome-disks`，先 umount，然后选择格式化，这时直接选择格式化为 Linux 的 Ext4，有一篇[图文介绍](https://hkgoldenmra.blogspot.com/2019/12/linux-luks-ext4.html)，不过没看时就已经自己操作了，只是让自己心安一下。

然后测试了一下读取速度，

```bash
~$ sudo hdparm -Tt /dev/sdc1

/dev/sdc1:
 Timing cached reads:   22298 MB in  1.99 seconds = 11228.47 MB/sec
 Timing buffered disk reads: 120 MB in  3.01 seconds =  39.89 MB/sec

~$ sudo hdparm -Tt /dev/sde1

/dev/sde1:
 Timing cached reads:   24390 MB in  1.99 seconds = 12281.26 MB/sec
 Timing buffered disk reads: 1318 MB in  3.00 seconds = 439.04 MB/sec
```

上面是普通的移动硬盘，底下是新买的移动 SSD 硬盘，差异还是很明显的。继续测试写入的速度，

```bash
~$ time dd if=/dev/zero of=/media/weiya/Extreme\ SSD/tempfile bs=1M count=1024
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 2.11846 s, 507 MB/s

real	0m2.131s
user	0m0.011s
sys	0m0.543s
~$ time dd if=/dev/zero of=/media/weiya/Seagate/tempfile bs=1M count=1024
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 12.4132 s, 86.5 MB/s

real	0m12.746s
user	0m0.000s
sys	0m1.551s
```

以及写出的速度，

```bash
~$ time dd if=/media/weiya/Extreme\ SSD/tempfile of=/dev/null bs=1M count=1024
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 4.01399 s, 268 MB/s

real	0m4.018s
user	0m0.000s
sys	0m0.442s
~$ time dd if=/media/weiya/Seagate/tempfile of=/dev/null bs=1M count=1024
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 65.6471 s, 16.4 MB/s

real	1m5.981s
user	0m0.010s
sys	0m0.650s
```

移动 SSD 硬盘完胜普通的移动硬盘。

参考链接：

- [在 Linux 上测试硬盘读写速度](http://einverne.github.io/post/2019/10/test-disk-write-and-read-speed-in-linux.html)

## 自动充放电

虽然一直知道插上电源充电会损耗电池容量，但是没想到竟然会损耗得那么严重，对于我正在使用的 ThinkPadT460P 来说，

```bash
~$ upower -i `upower -e | grep 'BAT'`
  native-path:          BAT0
  vendor:               SANYO
  model:                45N1767
  serial:               3701
  power supply:         yes
  updated:              Tue 01 Sep 2020 10:15:52 AM CST (106 seconds ago)
  has history:          yes
  has statistics:       yes
  battery
    present:             yes
    rechargeable:        yes
    state:               fully-charged
    warning-level:       none
    energy:              19.42 Wh
    energy-empty:        0 Wh
    energy-full:         19.58 Wh
    energy-full-design:  47.52 Wh
    energy-rate:         0 W
    voltage:             12.025 V
    percentage:          99%
    capacity:            41.2037%
    technology:          lithium-ion
    icon-name:          'battery-full-charged-symbolic'
```

现在的容量只有 41.2037%，一半都不到。心血来潮搜了下看看有没有什么软件能够支持自动充放电，竟然还真有，而且特别支持 ThinkPad 系列, [How can I limit battery charging to 80% capacity?](https://askubuntu.com/questions/34452/how-can-i-limit-battery-charging-to-80-capacity)

不过刚开始按照回答中的解决方案操作，最后 `sudo modprobe tp_smapi` 并不成功，大概是说没有这个 kernel 吧。不过因为这个回答挺早的，在评论中顺藤摸瓜找到针对更新版的 ThinkPad 的解决方案，[tlp for Ubuntu](https://linrunner.de/tlp/installation/ubuntu.html)

```bash
sudo add-apt-repository ppa:linrunner/tlp
sudo apt update
sudo apt install acpi-call-dkms tp-smapi-dkms
```

其中特别指出 `acpi-call-dkms` 用于 ThinkPads (X220/T420 and later)

然后查看

```bash
~$ sudo tlp-stat -b
--- TLP 1.3.1 --------------------------------------------

+++ Battery Features: Charge Thresholds and Recalibrate
natacpi    = inactive (no kernel support)
tpacpi-bat = active (thresholds, recalibrate)
tp-smapi   = inactive (ThinkPad not supported)
```

这时候按照 [Battery Charge Thresholds](https://linrunner.de/tlp/settings/battery.html) 修改 `/etc/tlp.conf`，并运行

```bash
sudo tlp start
```

但是似乎并没有起作用，仍然在充电，尝试拔了电源线来使之生效，但好像还是不行。总共有[三种生效方式](https://linrunner.de/tlp/settings/introduction.html#making-changes)，另外一种为重启。

猜测可能的原因是

> natacpi    = inactive (no kernel support)

但是发现 `natacpi` 只有 kernel 4.17 才开始支持，而当前我的 kernel 版本为

```bash
$ uname -r
4.15.0-112-generic
```

而且在 [Why is my battery charged up to 100% – ignoring the charge thresholds?](https://linrunner.de/tlp/faq/battery.html?highlight=natacpi#why-is-my-battery-charged-up-to-100-ignoring-the-charge-thresholds) 的
[ThinkPad T430(s)/T530/W530/X230 (and all later models)](https://linrunner.de/tlp/faq/battery.html?highlight=natacpi#thinkpad-t430-s-t530-w530-x230-and-all-later-models)
提到解决方案是

> Install a kernel ≥ 4.19 to make natacpi available

网上搜了一圈，发现更新内核还是有风险的，比如可能造成某些接口无法使用，这让我想起之前 wifi 接口搞不定的噩梦。那就先这样吧。

而且发现其实 [update & dist-upgrade](https://phoenixnap.com/kb/how-to-update-kernel-ubuntu) 可能还是会更新内核版本，但是不会更到最新？

!!! tip "upgrade vs dist-upgrade vs full-upgrade"
    参考 [What is “dist-upgrade” and why does it upgrade more than “upgrade”?](https://askubuntu.com/questions/81585/what-is-dist-upgrade-and-why-does-it-upgrade-more-than-upgrade)
    `upgrade` 只更新已经安装包的版本，不会额外下载包或卸载包
    `dist-upgrade` 会安装、卸载新包所依赖的包，而是更新内核版本也需要用它
    `full-upgrade`：不太清楚，试着运行完 dist-upgrade 后，再运行它，但是没反应。
    ```bash
    $ man apt-get
    ...
        upgrade
           upgrade is used to install the newest versions of all packages currently installed on the system from the sources enumerated in /etc/apt/sources.list. Packages currently
           installed with new versions available are retrieved and upgraded; under no circumstances are currently installed packages removed, or packages not already installed retrieved
           and installed. New versions of currently installed packages that cannot be upgraded without changing the install status of another package will be left at their current
           version. An update must be performed first so that apt-get knows that new versions of packages are available.

       dist-upgrade
           dist-upgrade in addition to performing the function of upgrade, also intelligently handles changing dependencies with new versions of packages; apt-get has a "smart" conflict
           resolution system, and it will attempt to upgrade the most important packages at the expense of less important ones if necessary. The dist-upgrade command may therefore
           remove some packages. The /etc/apt/sources.list file contains a list of locations from which to retrieve desired package files. See also apt_preferences(5) for a mechanism
           for overriding the general settings for individual packages.
    ```
    但是竟然没有看到 `full-upgrade`.

比如我发现 Ubuntu 18.04.5 LTS 实际上的内核版本应该是 5.0，甚至有 5.3，不过这似乎跟硬件有关，比如[这里](https://wiki.ubuntu.com/BionicBeaver/ReleaseNotes/ChangeSummary/18.04.5#Kernel_and_Hardware_support_updates)列了 `linux-aws-5.0`, `linux-aws-5.0`，不过我也看到了 `linux-gke-4.15`，所以还是不要乱升级的好，不然硬件不兼容又要继续折腾了。

话说回来，电池最后实在不行，就换了呗，反正这个是外置可拆卸的。

## scp a file with name including colon

add `./` before the file, since it will interpret colon `x:` as [user@]host prefix

refer to [How can I scp a file with a colon in the file name?](https://stackoverflow.com/questions/14718720/how-can-i-scp-a-file-with-a-colon-in-the-file-name)

## 添加虚拟内存

通过交换文件实现

```bash
# 创建大小为2G的文件swapfile
dd if=/dev/zero of=/mnt/swapfile bs=1M count=2048
# 格式化
mkswap /mnt/swapfile
# 挂载
swapon /mnt/swapfile
```

为了保证开机自动加载，在 `/etc/fstab` 加入

```bash
/mnt/swapfile swap swap defaults 0 0
```

具体每一列的含义可以通过 `man fstab` 查看。

挂载成功后就可以通过 `free -h` 查看内存情况。

参考 [Linux下如何添加虚拟内存](http://www.lining0806.com/linux%E4%B8%8B%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0%E8%99%9A%E6%8B%9F%E5%86%85%E5%AD%98/)

这个方法也可以解决 "virtual memory exhausted: Cannot allocate memory" 的问题。


## GPG error

```bash
$ sudo apt-get update
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/ InRelease: The following signatures were invalid: EXPKEYSIG 51716619E084DAB9 Michael Rutter <marutter@gmail.com>
W: Failed to fetch https://cloud.r-project.org/bin/linux/ubuntu/bionic-cran35/InRelease  The following signatures were invalid: EXPKEYSIG 51716619E084DAB9 Michael Rutter <marutter@gmail.com>
W: Some index files failed to download. They have been ignored, or old ones used instead.
```

and got the expired key via

```bash
$ apt-key list
pub   rsa2048 2010-10-19 [SCA] [expired: 2020-10-16]
...
uid           [ expired] Michael Rutter <marutter@gmail.com>
```

but it seems not work following [How to solve an expired key (KEYEXPIRED) with apt](https://linux-audit.com/how-to-solve-an-expired-key-keyexpired-with-apt/)

```bash
$ apt-key adv --keyserver keys.gnupg.net --recv-keys 51716619E084DAB9
Executing: /tmp/apt-key-gpghome.CYSI3C6heK/gpg.1.sh --keyserver keys.gnupg.net --recv-keys 51716619E084DAB9
gpg: key 51716619E084DAB9: "Michael Rutter <marutter@gmail.com>" not changed
gpg: Total number processed: 1
gpg:              unchanged: 1
```

then I tried another keyserver mentioned in [Installing R from CRAN Ubuntu repository: No Public Key Error](https://stackoverflow.com/questions/10255082/installing-r-from-cran-ubuntu-repository-no-public-key-error)

```bash
$ sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
[sudo] password for weiya:
Executing: /tmp/apt-key-gpghome.xUS3ZEg8N2/gpg.1.sh --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
gpg: key 51716619E084DAB9: "Michael Rutter <marutter@gmail.com>" 2 new signatures
gpg: Total number processed: 1
gpg:         new signatures: 2
```

Now, new signatures come, and no expired again.

Another one,

```bash
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04 ./ InRelease: The following signatures were invalid: EXPKEYSIG DFA175A75104960E Emulators OBS Project <Emulators@build.opensuse.org>
W: Failed to fetch https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/./InRelease  The following signatures were invalid: EXPKEYSIG DFA175A75104960E Emulators OBS Project <Emulators@build.opensuse.org>
W: Some index files failed to download. They have been ignored, or old ones used instead.
```

according to the record on [WeChat in Linux](software/#wechat-in-linux), it seems that this one is not important, and for simplest, I just untick this repository in the software setting.

## sftp via File Manager

在用 `connect to server` 时，经常弹出窗口要求输入用户名及密码，格式为 `sftp://xxx.xxx.xx.xx`，如果避免输入密码，不妨采用 `sftp://user@xxx.xxx.xx.xx`。不过有时在登录其它服务器时，不指定用户名还是直接登进去了，不太清楚为什么，猜想会不会是这几个服务器的用户名刚好跟本地相同。

## make the software searchable

If the software has `xx.destop` file, then

```bash
cp xx.destop ~/.local/share/applications
```

otherwise， create a `.desktop` file. More details refer to [How to pin Eclipse to the Unity launcher?](https://askubuntu.com/questions/80013/how-to-pin-eclipse-to-the-unity-launcher) and [How to add programs to the launcher (search)?](https://askubuntu.com/questions/285951/how-to-add-programs-to-the-launcher-search)

## Set printer for my laptop

1. choose LPD/LPR Host or Printer
2. set host as hpm605dn1.sta.cuhk.edu.hk

## MD5

```bash
~$ printf "hello\n" | md5sum
b1946ac92492d2347c6235b4d2611184  -
~$ printf "hello" | md5sum
5d41402abc4b2a76b9719d911017c592  -
~$ echo -n "hello" | md5sum
5d41402abc4b2a76b9719d911017c592  -
~$ echo "hello" | md5sum
b1946ac92492d2347c6235b4d2611184  -
```

where `-n` does not output the trailing newline `\n`, but 

```bash
~$ echo -n "hello\n" | md5sum
20e2ad363e7486d9351ee2ea407e3200  -
~$ echo -n "hello\n"
hello\n~$
```

other materals releated to MD5

- [三分钟学习 MD5](https://zhuanlan.zhihu.com/p/26592209)
- [为什么现在网上有很多软件可以破解MD5，但MD5还是很流行？](https://www.zhihu.com/question/22311285/answer/20960705)

## You have new mail

Here is a message when I login in to the Office PC,

```bash
You have new mail.
Last login: Thu May 20 13:29:14 2021 from 127.0.0.1
```

Refer to [“You have mail” – How to Read Mail in Linux Command Line](https://devanswers.co/you-have-mail-how-to-read-mail-in-ubuntu), the message is stored in the spool file, which is located at `/var/mail/$(whoami)`. The I found that this is a failed email when I wrote the mail notification script when there are new error message in `/var/log/apache2/error.log`.

## which vs type

在 CentOS7 服务器上，

```bash
$ which -v
GNU which v2.20, Copyright (C) 1999 - 2008 Carlo Wood.
GNU which comes with ABSOLUTELY NO WARRANTY;
This program is free software; your freedom to use, change
and distribute this program is protected by the GPL.
```

`which` 可以返回 alias 中的命令，而且更具体地，`man which` 显示可以通过选项 `--read-alias` 和 `--skip-alias` 来控制要不要包括 alias. 

而在本地 Ubuntu 18.04 机器上，不支持 `-v` 或 `--version` 来查看版本，而且 `man which` 也很简单，从中可以看出其大致版本信息，`29 Jun 2016`。

那怎么显示 alias 呢，[`type` 可以解决这个问题](https://askubuntu.com/questions/102093/how-to-see-the-command-attached-to-a-bash-alias)，注意查看其帮助文档需要用 `help` 而非 `man`。

```bash
$ type scp_to_chpc 
scp_to_chpc is a function
scp_to_chpc () 
{ 
    scp -r $1 user@host:~/$2
}
```

## systemd

[systemd](https://wiki.ubuntu.com/systemd) is a system and session manager for Linux, compatible with SysV and LSB init scripts. systemd 

- provides aggressive parallelization capabilities, 
- uses scoket and D-Bus activation for starting services, 
- offers on-demand starting of daemons,
- keeps track of processes using Linux cgroups
- supports snapshotting and restoring of the system state
- maintains mount and automount points
- implements an elaborate transactional dependency-based service control logic.

### control systemd once booted

The main command used to control systemd is `systemctl`. 

- `systemctl list-units`: list all units
- `systemctl start/stop [NAME]`: start/stop (activate/deactivate) one or more units
- `systemctl enable/disable [NAME]`: enable/disable one or more unit files
- `systemctl reboot`: shut down and reboot the system

## Custom Shortcut to Switch External Display Mode

办公室电脑既可以作为显示屏，也可以在 PC 模式下使用 Windows 系统。在 PC 模式下，在 Ubuntu 上通过 synergy 共享键鼠，但是此时存在一个问题，因为 HDMI 仍然连着，所以在移动鼠标时中间有个 gap，也就是需要跳过外接显示屏才能移动到 PC。

试过在 synergy 中将 PC 机设置为 T460p 上方，这样移动鼠标只需往上，不过体验不是很好，而且 Ubuntu 顶端有状态栏而 PC 端底部也有task bar，移动时能明显感受到延时。另外一个策略便是切换显示屏 mode，由 joint 模式切换成 mirror。

注意到，当处于 mirror 模式下，eDP-1-1 primary 显示为 `1920x1080+0+0`，而如果是 joint mode，尺寸为 `1920x1080+1920+0`。受 [Swap between monitor display modes using shortcut](https://askubuntu.com/questions/958914/swap-between-monitor-display-modes-using-shortcut)
 启发，决定写脚本自定义快捷键

```bash
~$ cat switch_mirror_joint.sh 
#!/bin/bash
currentmode=$(xrandr -q | grep "primary 1920x1080+0+0")
if [[ -n $currentmode ]]; then
    #echo "mirror"
    xrandr --output HDMI-1-1 --left-of eDP-1-1
else
    #echo "joint"
    xrandr --output HDMI-1-1 --same-as eDP-1-1
fi
```

然后进入 keyboard shortcut 设置界面，

- Name: `switch display mode`
- Command: `/home/weiya/switch_mirror_joint.sh`
- Shortcut: `Ctrl+F7`

之所以选择 `F7` 是因为本身 F7 也支持切换 display mode，但是默认 external monitor 在右侧。试图直接更改 F7 的 binding commands，相关的 Ubuntu 官方帮助文档 [Keyboard](https://help.ubuntu.com/stable/ubuntu-help/keyboard.html.en) 及配置文件 [Custom keyboard layout definitions](https://help.ubuntu.com/community/Custom%20keyboard%20layout%20definitions)，但是无从下手。

## monitor

- `ram_available`: percentage of estimated amount of RAM available for userspace processes, without causing swapping 
    - swap vs ram: see [深入理解swap交换分区理解及扩存 -- 知乎](https://zhuanlan.zhihu.com/p/201384108)
    - check with `free -m`
- `ram_in_use`: system memory utilization
- `30min_ram_swapped_out`: percentage of the system RAM swapped in the last 30 minutes  (???)
- `system.load`: 系统负载平均值（system load averages），它将正在运行的线程（任务）对系统的需求显示为平均运行数和等待线程数。Linux load averages 可以衡量任务对系统的需求，并且它可能大于系统当前正在处理的数量，大多数工具将其显示为三个平均值，分别为 1、5 和 15 分钟值（参考 [Linux Load Averages：什么是平均负载？ - 知乎](https://zhuanlan.zhihu.com/p/75975041)）。
    - `load_average_1`: system one-minute load average 
    - `load_average_5`: system five-minute load average 
    - ` load_average_15`: system fifteen-minute load average
    - 如果平均值为 0.0，意味着系统处于空闲状态
    - 如果 1min 平均值高于 5min 或 15min 平均值，则负载正在增加
    - 如果 1min 平均值低于 5min 或 15min 平均值，则负载正在减少
    - 如果它们高于系统 CPU 的数量，那么系统很可能会遇到性能问题
- `python.d_job_last_collected_secs`: number of seconds since the last successful data collection 
- `system.swap`
    - `used_swap`: swap memory utilization
- `system.cpu`
    - `10min_cpu_usage`: average CPU utilization over the last 10 minutes (excluding iowait, nice and steal) 
- `ipv4.udperrors`
    - `1m_ipv4_udp_receive_buffer_errors`： average number of UDP receive buffer errors over the last minute 
- `disk_space_usage`: disk / space utilization
- `linux_power_supply_capacity`: percentage of remaining power supply capacity
- `10s_ipv4_tcp_resets_received`: average number of received TCP RESETS over the last 10 seconds. This can be an indication that a service this host needs has crashed. Netdata will not send a clear notification for this alarm.