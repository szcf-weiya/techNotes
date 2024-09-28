---
comments: true
---

# Ubuntu

!!! info
    **Most notes on this page are based on my Ubuntu laptop.**
    
    - **20.04**: 2021-09-12 -> Now. [:link:](18to20.md)
    - **18.04**: 2020-04-12 -> 2021-09-12. [:link:](16to18.md)
    - **16.04**: ~ -> 2020-04-12
    - **14.04**: ~ -> ~

    Besides the sections listed on this page, the following blogs are organized along timeline,

    - [2024.03.09 CD Partition in USB Drive](blog/2024-03-09-CD-drive.md)
    - [2021.07.28 Use Old Kernel](blog/2021-07-28-use-old-kernel.md)
    - [2021.06.07 Switch Monitor Mode](blog/2021-06-07-switch-monitor.md)
    - [2020.09.01 Charge Battery Adaptively](blog/2020-09-01-charge-battery-adaptively.md)
    - [2020.08.25 Portable SSD](blog/2020-08-25-PSSD.md)


## Package Manager

### Advanced Package Tool (APT)

APT is a package management system for Debian and other Linux distributions based on it, such as Ubuntu.

#### PPA

> PPAs (Personal Package Archive) are repositories hosted on Launchpad. You can use PPAs to install or upgrade packages that are not available in the official Ubuntu repositories.

see also:

- [How do I resolve unmet dependencies after adding a PPA? - Ask Ubuntu](https://askubuntu.com/questions/140246/how-do-i-resolve-unmet-dependencies-after-adding-a-ppa)

!!! tip "proxy for apt"

    `proxychains` seems not work well before `sudo` or after `sudo`, and I don't want to add a system proxy permanently, then I found a temporary way,

    ```bash
    sudo http_proxy='http://user:pass@proxy.example.com:8080/' apt-get install package-name
    ```

    refer to [how to install packages with apt-get on a system connected via proxy?](https://askubuntu.com/questions/89437/how-to-install-packages-with-apt-get-on-a-system-connected-via-proxy)


### dpkg (Debian Package Manager)

"dpkg works under the hood of APT. While APT manages remote repositories and resolves dependencies for you, it use dpkg to actually make the changes of installing/removing package." [:link:](https://www.linuxfordevices.com/tutorials/debian/apt-vs-dpkg-debian)

- list installed packages: `dpkg -l`
    - flag 'ii' (installed) and 'rc' (removed but configuration), refer to [:link:](https://askubuntu.com/questions/18804/what-do-the-various-dpkg-flags-like-ii-rc-mean) for more details

### Snap

- `snaps`: the packages
- `snapd`: the tool for using packages

Snaps are self-contained applications running in a sandbox with mediated access to the host system. [:link:](https://en.wikipedia.org/wiki/Snap_(package_manager))

- list all installed packages: 

```bash
$ date
Sat 11 Jun 2022 08:22:05 PM CST
$ snap list
Name                             Version                     Rev    Tracking       Publisher   Notes
bare                             1.0                         5      latest/stable  canonical✓  base
canonical-livepatch              10.2.3                      146    latest/stable  canonical✓  -
core                             16-2.56                     13308  latest/stable  canonical✓  core
core18                           20220428                    2409   latest/stable  canonical✓  base
gnome-3-28-1804                  3.28.0-19-g98f9e67.98f9e67  161    latest/stable  canonical✓  -
gtk-common-themes                0.1-79-ga83e90c             1534   latest/stable  canonical✓  -
kde-frameworks-5                 5.47.0                      27     latest/stable  kde✓        -
kde-frameworks-5-core18          5.61.0                      32     latest/stable  kde✓        -
kde-frameworks-5-qt-5-14-core18  5.68.0                      4      latest/stable  kde✓        -
ksnip                            1.10.0                      443    latest/stable  dporobic    -
```

- check available updates

```bash
# without refreshing
$ snap refresh --list
# perform refreshing
$ sudo snap refresh
```

- check info of packages

```bash
$ snap info <snap name>
```

see also:

- [Ubuntu 推出的Snap应用架构有什么深远意义? -- 知乎](https://www.zhihu.com/question/47514122)

---

## Battery

!!! note "Charge Battery Adaptively"
    Check the [post](blog/2020-09-01-charge-battery-adaptively.md).

## Disk

??? note "Extend Disk"

    一直想扩容来着，但总是下不了决心。今天决定了，参考 google 搜索“Ubuntu 扩容”的前几条结果，便开始干了。

    1. 采用启动 U 盘，因为根目录在使用状态，幸好启动 U 盘还在。
    2. 使用 Gparted 时有个大大的 warning，说对含 /boot 分区的硬盘进行操作可能会不能正常启动，有点吓到了，最后还是狠下心继续下去了。
    3. 网上有人说，不要用 Gparted 对 Windows 进行压缩，而应该在 Windows 中进行压缩，可是此时已经开始了，想中断但怕造成更严重的后果，幸好最后启动 Windows 时只是多了步检查硬盘，并没有不能启动的状况。

    中间提心吊胆，好在最后顺利扩容完成。

    see also: [Why are there so many different ways to measure disk usage? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/120311/why-are-there-so-many-different-ways-to-measure-disk-usage)

!!! note "Portable SSD"
    Check the [post](blog/2020-08-25-PSSD.md)

!!! tip "Clean disk space"
    - autoremove `apt autoremove --purge`
    - manually find unused packages: `dpkg -l` and `apt list --installed`
    - for anaconda, run `conda clean -a`

!!! tip "Mount with Options"

    By default, the Segate disk would return owner as root.

    Use

    ```bash
    sudo umount /dev/XXX
    sudo mount -o rw,user,uid=1000,dmask=007,fmask=117 /dev/XXX /media/weiya/Segate
    ```

    where the first step might throw busy error, and the processes use the disk can be found via

    ```bash
    fuser -cu /local/mnt/
    ps -ef | grep XXX
    ```

    refer to

    - <https://askubuntu.com/questions/11840/how-do-i-use-chmod-on-an-ntfs-or-fat32-partition/956072>
    - <https://stackoverflow.com/questions/7878707/how-to-unmount-a-busy-device>

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

### sound device chooser

tried but not good. [Sound Input & Output Device Chooser](https://extensions.gnome.org/extension/906/sound-output-device-chooser/)

## Filesystem

### File Permissions

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

### Modify vs Change

在找学习资料时，突然不是很确定当初是否已经在用这台笔记本了，所以想确定一下本机的装机时间，参考 [How can I tell what date Ubuntu was installed?](https://askubuntu.com/questions/1352/how-can-i-tell-what-date-ubuntu-was-installed)，主要时通过查看文件的上次修改时间，比如

```bash
$ ls -lt /var/log/installer/
total 1200
-rw-rw-r-- 1 root   root 464905 Dec  2  2016 initial-status.gz
-rw-r--r-- 1 root   root     60 Dec  2  2016 media-info
-rw------- 1 syslog adm  334743 Dec  2  2016 syslog
-rw------- 1 root   root   2467 Dec  2  2016 debug
-rw------- 1 root   root 407422 Dec  2  2016 partman
-rw------- 1 root   root     17 Dec  2  2016 version
-rw------- 1 root   root    956 Dec  2  2016 casper.log
```

又如

```bash
$ ls -lt /
...
drwxrwxr-x   2 root root       4096 Dec  2  2016 cdrom
drwx------   2 root root      16384 Dec  2  2016 lost+found
drwxr-xr-x   2 root root       4096 Apr 21  2016 srv
```

出现了 2016.04.21 的一条记录。但如果我加上 `-c`，结果竟然不一样

```bash
$ ls -clt /
...
drwxrwxr-x   2 root root       4096 Dec  2  2016 cdrom
drwxr-xr-x   2 root root       4096 Dec  2  2016 srv
drwx------   2 root root      16384 Dec  2  2016 lost+found
```

难道 `ls` 默认显示的时间不是上次修改时间吗？？另外注意到 `srv` 其实是一个空文件夹。

这时我用 `stat` 进一步查看，

```bash
$ stat /srv
  File: /srv
  Size: 4096      	Blocks: 8          IO Block: 4096   directory
Device: 825h/2085d	Inode: 1179649     Links: 2
Access: (0755/drwxr-xr-x)  Uid: (    0/    root)   Gid: (    0/    root)
Access: 2021-05-05 08:43:20.955106697 +0800
Modify: 2016-04-21 06:07:49.000000000 +0800
Change: 2016-12-02 02:46:47.363728274 +0800
 Birth: -
```

发现有两个修改时间，`Modify` 和 `Change`，[两者区别:material-stack-overflow:](https://unix.stackexchange.com/questions/2464/timestamp-modification-time-and-created-time-of-a-file)在于

- `Modify`: the last time the file was modified (content has been modified)
- `Change`: the last time meta data of the file was changed (e.g. permissions)

然后进一步查看 Windows 系统的时间，

```bash
$ ll -clt
...
drwxrwxrwx  1 weiya weiya       4096 Oct  1  2016 '$Recycle.Bin'/
drwxrwxrwx  1 weiya weiya          0 Sep 29  2016  FFOutput/
-rwxrwxrwx  2 weiya weiya   15151172 Jul  2  2016  WindowsDENGL.tt2*
-rwxrwxrwx  2 weiya weiya   16092228 Jul  2  2016  WindowsDENG.tt2*
-rwxrwxrwx  2 weiya weiya   16217976 Jul  2  2016  WindowsDENGB.tt2*
-rwxrwxrwx  1 weiya weiya     400228 Mar 19  2016  bootmgr*
-rwxrwxrwx  1 weiya weiya          1 Mar 19  2016  BOOTNXT*
drwxrwxrwx  1 weiya weiya       8192 Mar 18  2016  Boot/
```

最早可以追溯到 2016.03.18.

### `.fuse_hidden`

I found a file `res/res_monodecomp/.fuse_hidden0016fbd000000001`, which is "a file was deleted but there is at least one software which is still using it, so it cannot be removed permanently" [:link:](https://askubuntu.com/questions/493198/what-is-a-fuse-hidden-file-and-why-do-they-exist). And it is suggested to use `lsof` to determine the application that uses such file. However,

```bash
$ lsof ./res/res_monodecomp/
lsof: WARNING: can't stat() tracefs file system /sys/kernel/debug/tracing
      Output information may be incomplete.
lsof: WARNING: can't stat() fuse.gvfsd-fuse file system /run/user/129/gvfs
      Output information may be incomplete.
```

### `/run/user/1000`

!!! info
    Post: 2022-04-05 09:58:32

Checking the usage of disk by `df -h`, here is a line,

```bash
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           2.0G   24K  2.0G   1% /run/user/129
tmpfs           2.0G  192K  2.0G   1% /run/user/1000
```

where 

- `tmpfs` (short for Temporary File System) is a temporary file storage paradigm implemented in many Unix-like operating systems. It is intended to appear as a mounted file system, but data is stored in volatile memory instead of a persistent storage device. [:link:](https://en.wikipedia.org/wiki/Tmpfs)
- `/run/user/$uid` is created by pam_systemd and used for storing files used by running processes for that user. [:link:](https://unix.stackexchange.com/questions/162900/what-is-this-folder-run-user-1000)

### 100% snap `/dev/loop`

```bash
$ df -h
/dev/loop1      9.0M  9.0M     0 100% /snap/canonical-livepatch/138
/dev/loop2      9.0M  9.0M     0 100% /snap/canonical-livepatch/146
/dev/loop3      114M  114M     0 100% /snap/core/13308
/dev/loop0      128K  128K     0 100% /snap/bare/5
```

"Having Snap images which consume 100% of their filesystem is perfectly acceptable" and "it's supposed to work that way", refer to [:link:](https://unix.stackexchange.com/questions/406534/snap-dev-loop-at-100-utilization-no-free-space)

See also: [What is /dev/loopx?](https://askubuntu.com/questions/906581/what-is-dev-loopx).

### Birth time

!!! info
    Post: 2023-03-06 17:23:57 -0500

有一篇博客写了草稿未 commit，然后匆忙更改后想知道上一次写作的时间。但是 `stat` 与 `ls` 中 Birth 字段为空。

首先尝试了 `debugfs`，但似乎这对于 ext4 有用。

[:link:](https://unix.stackexchange.com/questions/50177/birth-is-empty-on-ext4)

```bash
$ sudo debugfs -R 'stat 451072' /dev/sdc1
debugfs 1.45.5 (07-Jan-2020)
debugfs: Bad magic number in super-block while trying to open /dev/sdc1
/dev/sdc1 contains a ntfs file system labelled 'Seagate Backup Plus Drive'
stat: Filesystem not open
```

对于 ntfs，找到了[这个](https://unix.stackexchange.com/questions/87265/how-do-i-get-the-creation-date-of-a-file-on-an-ntfs-logical-volume),

```bash
$ getfattr --only-values -n system.ntfs_crtime_be four-generations-under-one-roof.md | 
    perl -MPOSIX -0777 -ne '$t = unpack("Q>"); print ctime $t/10000000-11644473600'
Sun Feb 26 22:52:09 2023
```

其中 `11644473600` 是 Windows 的开始计时时间点 (1601-01-01T00:00:00Z) 与 Linux 计时开始点 (1970-01-01T00:00:00Z) 所差的秒数（[:link:](https://stackoverflow.com/questions/6161776/convert-windows-filetime-to-second-in-unix-linux)）。

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

??? note "霞鹜文楷"
    <https://github.com/lxgw/LxgwWenKai>

    ```bash
    git clone git@github.com:lxgw/LxgwWenKai.git
    cp LxgwWenKai/fonts/TTF/* ~/.local/share/fonts/
    ```

??? note "仿宋 & 黑体"

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

    **NOTE:** 除了系统的字体目录，也可以选择用户目录，"~/.local/share/fonts/". 另见 [:link:](https://askubuntu.com/questions/191778/how-to-install-many-font-files-quickly-and-easily) 另外 `fc-cache` 也不是必须的。

    安装完成后，

    ```bash
    $ fc-list :lang=zh
    /usr/share/fonts/truetype/custom/simhei.ttf: SimHei,黑体:style=Regular,Normal,obyčejné,Standard,Κανονικά,Normaali,Normál,Normale,Standaard,Normalny,Обычный,Normálne,Navadno,Arrunta
    /usr/share/fonts/truetype/custom/STFANGSO.TTF: STFangsong,华文仿宋:style=Regular
    ```

!!! tip "Font Name in xeCJK"
    在 `\setCJKmainfont{}` 中使用 `fc-list` 冒号之前的字体名。也可以使用字体文件，但需要指定具体路径，否则只会在当前目录下寻找。

### Some Free Fonts

- [Mozilla's Fira Type Family](https://github.com/mozilla/Fira)
    - [Fira for Metropolis theme](https://github.com/matze/mtheme/issues/280)
    - [Fira Code](https://github.com/tonsky/FiraCode)
        - [知乎：Fira Code —— 专为编程而生的字体](https://zhuanlan.zhihu.com/p/65362086)

## Headphone

!!! tip "restart sound"
    ```r
    pulseaudio -k && sudo alsa force-reload
    ```

The output candidates are

![image](https://user-images.githubusercontent.com/13688320/130001206-2b37623c-e461-41ef-b8c4-721784d5da22.png)

- HDMI/DisplayPort - Built-in Audio
- Speakers - Built-in Audio
- Headphone - LE-Bose QC35 II

and there are two profiles for `Headphone`

- High Fidelity Playback (A2DP Sink)
- Headset Head Unit (HSP/HFP)

also two profiles for `Speakers`

- Analog Stereo Output
- Analog Surround 4.0 Output

and one profile for `HDMI/DisplayPort`

- Digital Stereo (HDMI) Output

There are two devices for sound input

![image](https://user-images.githubusercontent.com/13688320/130001212-a75162b1-a221-4e4b-b719-65b611c06c56.png)

- Internal Microphone - Built-in Audio
- Bluetooth Input - LE-Bose QC35 II

Here are some observations

- The profile `High Fidelity Playback` is much more confortable than another profile.
- If `Output = Headphone + High Fidelity`, then `Input` cannot be set as `Bluetooth Input`
- Otherwise, if set `Input` as `Bluetooth Input`, then the Profile of `Output` would become to `Headset Head Unit`

Sometimes, no sound from the headphone, and possibly the headphone is connected to another device. Refer to [Using multiple Bluetooth® connections](https://www.bose.com/en_us/support/articles/HC1283/productCodes/qc35_ii/article.html) for switching the connected devices for the headphone. 

> When two devices are actively connected, you can play music from either device. To switch between connected devices, pause the first device and play music from the second.

But sometimes either devices are playing sound, a less elegant way is to disconnect other devices.

see also [Pair Bose QuietComfort 35 with Ubuntu over Bluetooth - Ask Ubuntu](https://askubuntu.com/questions/833322/pair-bose-quietcomfort-35-with-ubuntu-over-bluetooth)

!!! tip "Turn on/off Bluebooth from CLI"
    某次，合上笔记本盖子然后打开后，找不到蓝牙，而且在设置界面中也无法打开蓝牙。
    ```bash
    rfkill block bluetooth
    rfkill unblock bluetooth
    ```
    参考 [:link:](https://askubuntu.com/questions/380096/turn-on-off-bluetooth-from-shell-not-from-applet)


!!! note "always a2dp_sink!"
    Sometimes it is noisy, and annoying "call from". This is due to the profile, but using the GUI to change it seems not work. So try to work from the command line.
    ```bash
    # list available cards
    $ pacmd list-cards
    	profiles:
            headset_head_unit: Headset Head Unit (HSP/HFP) (priority 30, available: unknown)
            a2dp_sink: High Fidelity Playback (A2DP Sink) (priority 40, available: yes)
            off: Off (priority 0, available: yes)
        active profile: <a2dp_sink>
    ``` 
    if the active profile is `headset_head_unit`, it would be noisy. Tried

    ```bash
    pacmd set-card-profile <card name> a2dp_sink
    ```
    but it said "failed to set".

    However, after reconnecting, it works well, and it can easily switch between `headset_head_unit` and `a2dp_sink`.

    See also: [:link:](https://www.reddit.com/r/archlinux/comments/8hzylp/annoying_call_from_message_when_connecting_bose/) and [:link:](https://unix.stackexchange.com/questions/462670/set-default-profile-for-pulseaudio)


## Nvidia Driver

Install via the GUI `Software & Updates`. If succeed, then

```bash
$ nvidia-smi
```

can display the GPU memory usage, together with the versions of driver and CUDA,

```bash
$ nvidia-smi 
Mon Aug  2 22:08:19 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce 940MX       Off  | 00000000:02:00.0 Off |                  N/A |
| N/A   63C    P0    N/A /  N/A |    724MiB /  2004MiB |      9%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       400      G   WeChatWeb.exe                       9MiB |
|    0   N/A  N/A       663      G   ...cent\WeChat\WeChatApp.exe        7MiB |
|    0   N/A  N/A      4454      G   ...AAAAAAAAA= --shared-files       59MiB |
|    0   N/A  N/A      7440      G   /usr/lib/xorg/Xorg                437MiB |
```

refer to [Linux安装NVIDIA显卡驱动的正确姿势](https://blog.csdn.net/wf19930209/article/details/81877822) for other approaches (seems more technical).

!!! note "Different CUDA version shown by nvcc and NVIDIA-smi"

    refer to [Different CUDA versions shown by nvcc and NVIDIA-smi](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi)

    > CUDA has 2 primary APIs, the runtime and the driver API. Both have a corresponding version
    >
    > - The necessary support for the driver API (e.g. libcuda.so on linux) is installed by the GPU driver installer.
    > - The necessary support for the runtime API (e.g. libcudart.so on linux, and also nvcc) is installed by the CUDA toolkit installer (which may also have a GPU driver installer bundled in it).

    `nvidia-smi`: installed by the GPU driver installer, and generally has the GPU driver in view, not anything installed by the CUDA toolkit installer.
    `nvcc`: the CUDA compiler-driver tool that is installed with the CUDA toolkit, will always report the CUDA runtime version that it was built to recognize.


## Kernel

- [Go back to old kernel](blog/2021-07-28-use-old-kernel.md)

## Printer

### Use Department Printer

The department printer is only accessible from the office PC, but request to switch the cable from office PC to my laptop, I can configure to use the printer from my laptop. The steps are

1. choose LPD/LPR Host or Printer
2. set host as `hpm605dn1.sta.cuhk.edu.hk`

The detailed settings are

![image](https://user-images.githubusercontent.com/13688320/130003632-67fa7b32-8aad-4c83-a2f0-8291bb1b942c.png)

finally, the test page can be successfully printed,

![image](https://user-images.githubusercontent.com/13688320/130003854-9e434caf-211a-403b-a9ff-f709f38f07cf.png)

If we forget the hostname, actually we can get the ip address from the panel of printer,

step 1 | step 2
--| --
![image](https://user-images.githubusercontent.com/13688320/130004107-199b504c-9216-4e97-95b7-39287c5e2f68.png) | ![image](https://user-images.githubusercontent.com/13688320/130004117-cf2790f4-247f-420b-8e77-7739df41b14d.png)

We can validate the hostname and the ip point to the same machine,

```bash
weiya@stapc220:~$ ping hpm605dn1.sta.cuhk.edu.hk
PING hpm605dn1.sta.cuhk.edu.hk (172.16.37.238) 56(84) bytes of data.
```

### Share Printer

现有台 HP-Deskjet-1050-J410-series 打印机，通过 USB 接口。直接连接在 Ubuntu 上是可以实现打印功能的，现在想贡献给局域网内的其他设备，参考 [使用Linux共享打印机](https://www.jianshu.com/p/a1c4fc6d9ce8)，主要步骤为

1. 安装 CUPS 服务，`sudo apt-get install cups` 并启动，`sudo service cups start`
2. 在 `127.0.0.1:631` 的 `Administration >> Advanced` 勾选 `Allow printing from the Internet`，并保存。
3. 打开防火墙，`sudo ufw allow 631/tcp`

在同一局域网内的 Windows 设备中，添加该打印机，地址即为Ubuntu中浏览器的地址，注意将 `127.0.0.1` 换成局域网 ip。如果顺利的话，添加后需要添加驱动程序，可以在 HP 官网下载。

## Add Virtual Memory

通过交换文件实现

```bash
# 创建大小为2G的文件swapfile
dd if=/dev/zero of=/mnt/swapfile bs=1M count=2048
# 格式化
mkswap /mnt/swapfile
# 挂载
swapon /mnt/swapfile
```

注意文件应为 `root:root`，否则会提示

> insecure file owner 1000, 0 (root) suggested.

另见 [How to Resolve the Insecure warning in Swapon?](https://unix.stackexchange.com/questions/297149/how-to-resolve-the-insecure-warning-in-swapon)

为了保证开机自动加载，在 `/etc/fstab` 加入

```bash
/mnt/swapfile swap swap defaults 0 0
```

具体每一列的含义可以通过 `man fstab` 查看。

挂载成功后就可以通过 `free -h` 查看内存情况。

!!! warning "swapfile on PSSD"
	如果 swapfile 在外接移动硬盘上，则开机时需要提前插好外接硬盘。否则会卡在开机那一步。反之，如果为了追求便携性不依赖于移动硬盘，则最后不要在移动硬盘中创建 swapfile.

参考 [Linux下如何添加虚拟内存](http://www.lining0806.com/linux%E4%B8%8B%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0%E8%99%9A%E6%8B%9F%E5%86%85%E5%AD%98/)

这个方法也可以解决 "virtual memory exhausted: Cannot allocate memory" 的问题。

调整 swapiness，默认值为 60，

```bash
$ cat /proc/sys/vm/swappiness
```

越高表示越积极使用 swap 空间。

临时性使用

```bash
$ sudo sysctl vm.swappiness=80
```

参考 [linux系统swappiness参数在内存与交换分区间优化](http://blog.itpub.net/29371470/viewspace-1250975)

---

## WiFi

!!! info
    Post: 2022-04-01 23:57:09

校园网突然无法打开微信图片，公众号文章也无法加载，无法 ping 通 `mp.weixin.qq.com`，但在服务器上可以。所以第一个自然想法是利用动态转发搭建隧道，即在本地端运行 

```bash
$ ssh -D 30002 server
```

然后便可以通过 socks5://127.0.0.1:30002 进行代理。加了代理之后，首先能够在浏览器端打开公众号文章。但想要微信本身也进行代理，并不直接，可能因为 wine 套壳的原因，后来也没有继续细究。

另一种方式便是换 wifi。除了 CUHK1x，也可以使用 eduroam，点击连接时，竟然发现连接状态下的 security 是哈佛帐号，然而我并没有利用哈佛帐号连接过 eduroam，而且帐号早已失效。其原因，很可能就是因为当时连接 Harvard Secure 时，会下载一个 `JoinNow`，并在本地运行 [:link:](https://harvard.service-now.com/ithelp?id=kb_article&sys_id=8720ee5c0fb0fe802dfe5bd692050eef#LinuxSecure)。

而此时所连接的 eduroam 中 security 便指向 joinnow 的两个密钥文件，

```bash
~/.joinnow/tls-client-certs$ ls
sw2-joinnow-client-cert-**************.crt
sw2-joinnow-client-cert-**************.p12
```

不过在该 wifi 下，不能直接访问服务器。于是想切换到学校帐号的 eduroam，但又想保留当前 profile。所以尝试直接复制 wifi profile 文件，

```bash
/etc/NetworkManager/system-connections$ sudo cp A B
```

然后修改每个 profile 的细节，注意 uuid 也必须修改，运行 `uuid` 生成，不然会视为同一个连接。修改完成后重启网络服务，

```bash
sudo systemctl restart NetworkManager
```

参考 [:link:](https://askubuntu.com/questions/936817/can-i-create-two-different-profiles-for-one-wifi-network)

另外，命令行操作进行网络连接详见 <https://www.makeuseof.com/connect-to-wifi-with-nmcli/>

### Delete Hotspot

升级到 Ubuntu 18.04 后，开机自动连接到 Hotspot，每次需要手动禁止并改成 Wifi 连接，这个可以直接删除保存好的 Hotspot 连接

```bash
cd /etc/NetworkManager/system-connections/
sudo rm Hotspot
```

参考 [How to remove access point from saved list](https://askubuntu.com/questions/120415/how-to-remove-access-point-from-saved-list/120447)

### WiFi Hotpot (16.04)

Refer to

1. [3 Ways to Create Wifi Hotspot in Ubuntu 14.04 (Android Support)](http://ubuntuhandbook.org/index.php/2014/09/3-ways-create-wifi-hotspot-ubuntu/)
2. [How do I create a WiFi hotspot sharing wireless internet connection (single adapter)?](https://askubuntu.com/questions/318973/how-do-i-create-a-wifi-hotspot-sharing-wireless-internet-connection-single-adap)

几处不同：

1. 选择 `mode` 时，直接选择 `hotpot` 即可，后面也无需更改文件
2. 设置密码时位数不能少于 8 位
3. 连接 WiFi 时 似乎需要 enable wifi。

## Keyboard

实验室有一支 IKBC CD 87T 蓝牙键盘，于是便想试试。

- 连接：详见[说明书](manual_IKBC.pdf)
- Win 键失效：尝试按 `Fn + Right Win` 及左 `Fn + Left Win`，参考 [IKBC键盘Win键失效的解决办法](https://blog.csdn.net/norman_irsa/article/details/114735798)

## 1m_ipv4_udp_receive_buffer_errors

!!! info
    The raw records can be found [here](https://github.com/szcf-weiya/techNotes/issues/32#issuecomment-899313680).

After replacing one of 4GB RAM with 16GB one, it throws 

> 1m_ipv4_udp_receive_buffer_errors

frequently, such as

![](https://user-images.githubusercontent.com/13688320/129532171-f1c108dd-f4f8-44a8-a062-96e2ae8441a3.png)

Following the instructions

- [linux 系统 UDP 丢包问题分析思路 | Cizixs Write Here](https://cizixs.com/2018/01/13/linux-udp-packet-drop-debug/)
- [netdata ipv4 UDP errors - Server Fault](https://serverfault.com/questions/899364/netdata-ipv4-udp-errors)

but no `drops`

```bash
wlp3s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.13.59.193  netmask 255.255.128.0  broadcast 10.13.127.255
        inet6 fe80::def:d34:a2c:da88  prefixlen 64  scopeid 0x20<link>
        ether a4:34:d9:e8:9a:bd  txqueuelen 1000  (Ethernet)
        RX packets 17556713  bytes 20549547923 (20.5 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 14008823  bytes 17151970123 (17.1 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

then try to enlarge `net.core.rmem_max` and `net.core.rmem_default`. The current values are

```bash
~$ sysctl net.core | grep mem
net.core.optmem_max = 20480
net.core.rmem_default = 212992
net.core.rmem_max = 212992
net.core.wmem_default = 212992
net.core.wmem_max = 212992
```

the update it via

```bash
~$ sudo sysctl -w net.core.rmem_default=1048576
net.core.rmem_default = 1048576
~$ sudo sysctl -w net.core.rmem_max=2097152
net.core.rmem_max = 2097152
~$ sysctl net.core | grep mem
net.core.optmem_max = 20480
net.core.rmem_default = 1048576
net.core.rmem_max = 2097152
net.core.wmem_default = 212992
net.core.wmem_max = 212992
```

it seems to work since much fewer warning message of `1m_ipv4_udp_receive_buffer_errors`

## 开机自启动

搜索 `Startup` 便可弹出开机自启动软件界面，

![Selection_2329](https://user-images.githubusercontent.com/13688320/133000670-f1e9062e-8ba3-45b1-87c6-5b5e89d5150e.png)

## Connect Android Phones

If the computer does not recognize the android devices, try to use the original cable. See [:link:](https://askubuntu.com/questions/518479/ubuntu-doesnt-recognize-android-devices-anymore) for other possibilities.
