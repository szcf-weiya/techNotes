# Ubuntu

!!! info
    **Most notes on this page are based on my Ubuntu laptop.**
    
    - 20.04: 2021-09-12 -> Now. [:link:](18to20.md)
    - 18.04: 2020-04-12 -> 2021-09-12. [:link:](16to18.md)
    - 16.04: ~ -> 2020-04-12
    - 14.04: ~ -> ~

## Package Manager

### Advanced Package Tool (APT)

APT is a package management system for Debian and other Linux distributions based on it, such as Ubuntu.

#### PPA

> PPAs (Personal Package Archive) are repositories hosted on Launchpad. You can use PPAs to install or upgrade packages that are not available in the official Ubuntu repositories.

see also:

- [How do I resolve unmet dependencies after adding a PPA? - Ask Ubuntu](https://askubuntu.com/questions/140246/how-do-i-resolve-unmet-dependencies-after-adding-a-ppa)

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

## Sound with Headphone

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

There are two devices fro sound input

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

## Use Department Printer

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

## Share Printer

现有台 HP-Deskjet-1050-J410-series 打印机，通过 USB 接口。直接连接在 Ubuntu 上是可以实现打印功能的，现在想贡献给局域网内的其他设备，参考 [使用Linux共享打印机](https://www.jianshu.com/p/a1c4fc6d9ce8)，主要步骤为

1. 安装 CUPS 服务，`sudo apt-get install cups` 并启动，`sudo service cups start`
2. 在 `127.0.0.1:631` 的 `Administration >> Advanced` 勾选 `Allow printing from the Internet`，并保存。
3. 打开防火墙，`sudo ufw allow 631/tcp`

在同一局域网内的 Windows 设备中，添加该打印机，地址即为Ubuntu中浏览器的地址，注意将 `127.0.0.1` 换成局域网 ip。如果顺利的话，添加后需要添加驱动程序，可以在 HP 官网下载。

## Portable SSD

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

## Charge Battery Adaptively

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


## Extend Disk

一直想扩容来着，但总是下不了决心。今天决定了，参考 google 搜索“Ubuntu 扩容”的前几条结果，便开始干了。

1. 采用启动 U 盘，因为根目录在使用状态，幸好启动 U 盘还在。
2. 使用 Gparted 时有个大大的 warning，说对含 /boot 分区的硬盘进行操作可能会不能正常启动，有点吓到了，最后还是狠下心继续下去了。
3. 网上有人说，不要用 Gparted 对 Windows 进行压缩，而应该在 Windows 中进行压缩，可是此时已经开始了，想中断但怕造成更严重的后果，幸好最后启动 Windows 时只是多了步检查硬盘，并没有不能启动的状况。

中间提心吊胆，好在最后顺利扩容完成。

see also: [Why are there so many different ways to measure disk usage? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/120311/why-are-there-so-many-different-ways-to-measure-disk-usage)

## Rename Portable Disk

终端输入

```bash
gnome-disks
```

在设置齿轮图标中选择 `Edit Mount Options`，修改 `Mount Point`。注意重新挂载后才能生效。

详见[How to change hard drive name](https://askubuntu.com/questions/904561/how-to-change-hard-drive-name/904564)

## Use Old Kernel

最近几天，T460P 经常自动重启，而且往往重启前花屏，甚至今天带着电脑去找老板，也重启了两次。在 `/var/crash` 目录下能发现 crash 报告，但是并不知道怎么使用

```bash
drwxr-sr-x  2 root     whoopsie     4096 Jul 22 20:10 202107222000/
drwxr-sr-x  2 root     whoopsie     4096 Jul 23 05:40 202107230539/
drwxr-sr-x  2 root     whoopsie     4096 Jul 23 08:54 202107230854/
drwxr-sr-x  2 root     whoopsie     4096 Jul 24 14:23 202107241422/
drwxr-sr-x  2 root     whoopsie     4096 Jul 26 09:03 202107260517/
drwxr-sr-x  2 root     whoopsie     4096 Jul 27 09:26 202107270926/
drwxr-sr-x  2 root     whoopsie     4096 Jul 27 23:09 202107272308/
drwxr-sr-x  2 root     whoopsie     4096 Jul 28 09:41 202107280940/
drwxr-sr-x  2 root     whoopsie     4096 Jul 28 09:44 202107280944/
drwxr-sr-x  2 root     whoopsie     4096 Jul 28 10:14 202107281014/
drwxr-sr-x  2 root     whoopsie     4096 Jul 28 11:05 202107281105/
drwxr-sr-x  2 root     whoopsie     4096 Jul 28 11:48 202107281148/
-rw-r--r--  1 kernoops whoopsie     2249 Jul 24 14:40 linux-image-4.15.0-151-generic.183202.crash
-rw-r-----  1 root     whoopsie    34940 Jul 22 20:01 linux-image-4.15.0-151-generic-202107222000.crash
-rw-r-----  1 root     whoopsie    29062 Jul 23 08:55 linux-image-4.15.0-151-generic-202107230854.crash
-rw-r-----  1 root     whoopsie    34532 Jul 24 14:23 linux-image-4.15.0-151-generic-202107241422.crash
-rw-r-----  1 root     whoopsie    35752 Jul 26 08:56 linux-image-4.15.0-151-generic-202107260517.crash
-rw-r--r--  1 kernoops whoopsie      746 Jul 28 09:40 linux-image-4.15.0-151-generic.30586.crash
-rw-r--r--  1 kernoops whoopsie      752 Jul 28 09:40 linux-image-4.15.0-151-generic.30735.crash
```

其中每个文件夹中有

```bash
$ ll 202107281148
total 109160
-rw-------  1 root whoopsie     78507 Jul 28 11:48 dmesg.202107281148
-rw-------  1 root whoopsie 111683440 Jul 28 11:48 dump.202107281148
```

联想下最近的操作，很有可能是通过 `apt upgrade` 更新了次内核，查看具体细节

```bash
$ vi /var/log/apt/history.log
Start-Date: 2021-07-21  20:46:59
Commandline: aptdaemon role='role-commit-packages' sender=':1.1117'
Install: linux-headers-4.15.0-151-generic:amd64 (4.15.0-151.157, automatic), linux-modules-4.15.0-151-generic:amd64 (4.15.0-151.157, automatic), linux-image-4.15.0-151-generic:amd64 (4.15.0-151.157, automatic), linux-modules-extra-4.15.0-151-generic:amd64 (4.15.0-151.157, automatic), linux-headers-4.15.0-151:amd64 (4.15.0-151.157, automatic)
Upgrade: libnvidia-gl-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-gl-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), linux-headers-generic:amd64 (4.15.0.147.134, 4.15.0.151.139), linux-libc-dev:amd64 (4.15.0-147.151, 4.15.0-151.157), linux-crashdump:amd64 (4.15.0.147.134, 4.15.0.151.139), libsystemd0:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), libsystemd0:i386 (237-3ubuntu10.48, 237-3ubuntu10.49), linux-image-generic:amd64 (4.15.0.147.134, 4.15.0.151.139), nvidia-driver-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), containerd:amd64 (1.5.2-0ubuntu1~18.04.1, 1.5.2-0ubuntu1~18.04.2), nvidia-kernel-source-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-fbc1-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-fbc1-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), nvidia-dkms-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), google-chrome-stable:amd64 (91.0.4472.164-1, 92.0.4515.107-1), nvidia-utils-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-decode-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-decode-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), udev:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), nvidia-kernel-common-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), typora:amd64 (0.10.11-1, 0.11.0-1), xserver-xorg-video-nvidia-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-encode-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-encode-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), nvidia-compute-utils-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), initramfs-tools-bin:amd64 (0.130ubuntu3.12, 0.130ubuntu3.13), linux-signed-generic:amd64 (4.15.0.147.134, 4.15.0.151.139), libudev1:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), libudev1:i386 (237-3ubuntu10.48, 237-3ubuntu10.49), libnvidia-ifr1-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-ifr1-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-common-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnss-myhostname:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), libnvidia-cfg1-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), systemd-sysv:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), libpam-systemd:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), systemd:amd64 (237-3ubuntu10.48, 237-3ubuntu10.49), libnvidia-extra-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), linux-generic:amd64 (4.15.0.147.134, 4.15.0.151.139), initramfs-tools-core:amd64 (0.130ubuntu3.12, 0.130ubuntu3.13), initramfs-tools:amd64 (0.130ubuntu3.12, 0.130ubuntu3.13), libnvidia-compute-460-server:amd64 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1), libnvidia-compute-460-server:i386 (460.73.01-0ubuntu0.18.04.1, 460.91.03-0ubuntu0.18.04.1)
End-Date: 2021-07-21  20:51:53

Start-Date: 2021-07-22  06:31:33
Commandline: /usr/bin/unattended-upgrade
Remove: linux-headers-4.15.0-144:amd64 (4.15.0-144.148), linux-headers-4.15.0-144-generic:amd64 (4.15.0-144.148)
End-Date: 2021-07-22  06:31:36
```

!!! tip
    对于 `.gz` 的日志文件，可以使用 `zcat` 直接查看。
    另外注意，`file.gz` 解压时默认不会保存原始文件，或者指定 `gunzip -k` 选项，或者
    ```bash
    gunzip < file.gz > file
    ```
    详见 [Unzipping a .gz file without removing the gzipped file](https://unix.stackexchange.com/questions/156261/unzipping-a-gz-file-without-removing-the-gzipped-file)

发现其更新时间恰恰在诸多 crash 的前一天。所以试图切换到老的内核版本。

注意到在启动时，第二个选项是 `Advanced options for Ubuntu`，点进去有若干个内核，我的是

```bash
4.15.0-151
4.15.0-151 (recovery mode)
4.15.0-147
4.15.0-147 (recovery mode)
```

选择 `4.15.0-147`，进去之后发现扩展屏幕无法识别，另外 Nvidia 也没有识别。于是试着重装 nvidia driver，这次选择的是 `nvidia-driver-460-server`，这也是[此前使用的版本](https://github.com/szcf-weiya/techNotes/issues/11#issuecomment-885333581)。

然后重启。

重启时注意还是要选择 `4.15.0-147`，不如默认进的还是 `151`。

中间误入了一次 `151`，结果还没登录就自动重启了，然后乖乖选择 `147`，但是这次还是没能识别 HDMI，不过 nvidia-smi 以及 nvidia-settings 都正常，`xrandr` 输出也是没有识别出显示屏。

试着继续重启，这次竟然可以了！

后面在 nvidia-settings 中试了将 `PRIME profiles` 由 `NVIDIA (performance mode)` 改成 `NVIDIA on-demand`,然后重启，结果竟然进不去系统了，每次输入密码回车都要求重新输入密码。

原因很可能是 PRIME 的这一改动，于是进入 tty 模式，通过命令行改回来

```bash
# 查询当前选择
$ prime-select query
# 切换至 nvidia
$ prime-select nvidia
# 切换至 intel
$ prime-select intel
```

然后重启终于恢复正常。

注意现在每次重启都需要选择内核版本，一个自然想法是修改默认内核。参考 [How Do I Change the Default Boot Kernel in Ubuntu?](https://support.huaweicloud.com/intl/en-us/trouble-ecs/ecs_trouble_0327.html)

将 `/etc/default/grub` 中 `GRUB_DEFAULT` 修改至 `1>2`，其中

- `1` 代表 `Advanced options for Ubuntu` 的顺序，因其在第二位，顺序从 0 算起，默认值就是 0
- `>2` 代表在子目录下位于第 2 位（顺序从 0 算起，即第三个）

重启之前需要运行

```bash
$ sudo update-grub
```

重启之后可以看到自动选择了 `1>2`。 

### disable update of kernel

今天竟然又重启了，有点无法理解。第一次怀疑是温度过高，因为 `syslog` 中重启前的一条记录信息为

```bash
Device: /dev/sdc [SAT], SMART Usage Attribute: 194 Temperature_Celsius changed from 60 to 61
```

然后第二次竟然发现 crash 报告竟然是 `151`，明明应该选择了第二个 147 呀。

```bash
$ ls /var/crash
linux-image-4.15.0-151-generic-202108070025.crash
linux-image-4.15.0-151-generic-202108071819.crash
```

重启时才发现竟然多了个 kernel 153！！于是第二个变成了 151，第三个才是 147。

虽然将启动选择再改成第三个应该也是可以的，但是万一以后又多了一个呢，顺序岂不又变了，所以更好的方法便是禁止 kernel 更新。另外将最新的 151 删掉。

- 删掉 kernel

查看安装 kernel 的完整名称，

```bash
$ dpkg -l | grep linux-image | grep "^ii"
```

然后进行删除

```bash
$ sudo apt purge linux-image-4.15.0-153-generic
```

再次重启就会发现第一项 153 消失了。

refer to [Ubuntu 18.04 remove all unused old kernels](https://www.cyberciti.biz/faq/ubuntu-18-04-remove-all-unused-old-kernels/)

- 禁止 kernel 更新

参考 [How to I prevent Ubuntu from kernel version upgrade and notification?](https://askubuntu.com/questions/938494/how-to-i-prevent-ubuntu-from-kernel-version-upgrade-and-notification)

```bash
sudo apt-mark hold 4.15.0-147-generic
```

## Default Software

网页文件 `.html` 默认用百度网盘打开，之前通过 `KDE System Setting` 修改了默认软件，

![](https://user-images.githubusercontent.com/13688320/117541554-fb955800-b046-11eb-8577-f39fdbf406bc.png)

但似乎并没有解决问题。

试着参考 [Open files with other applications](https://help.ubuntu.com/stable/ubuntu-help/files-open.html.en) 的步骤进行设置

- 右键选择 `Properties`
- 然后选择 `Open With`
- 选择特定软件，`Set as default`

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

## ubuntu 连接 sftp 服务器

参考[Use “Connect to Server” to connect to SFTP](https://askubuntu.com/questions/349873/use-connect-to-server-to-connect-to-sftp)

## Ubuntu的回收站

参考 [https://blog.csdn.net/DSLZTX/article/details/46685959](https://blog.csdn.net/DSLZTX/article/details/46685959)

## install win on ubuntu

参考[http://www.linuxdeveloper.space/install-windows-after-linux/](http://www.linuxdeveloper.space/install-windows-after-linux/)

## 开机自启动

搜索 `Startup` 便可弹出开机自启动软件界面，

![Selection_2329](https://user-images.githubusercontent.com/13688320/133000670-f1e9062e-8ba3-45b1-87c6-5b5e89d5150e.png)

## Mount with Options

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

- <https://askubuntu.com/questions/11840/how-do-i-use-chmod-on-an-ntfs-or-fat32-partition/956072#956072>
- <https://stackoverflow.com/questions/7878707/how-to-unmount-a-busy-device>
