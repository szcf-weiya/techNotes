---
comments: true
---

# Use Old Kernel

!!! info
    Post: [2021.07.28](https://github.com/szcf-weiya/techNotes/commit/7ceeacf682da9d96b8e4e97e09d218c654d6f9b5)

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

!!! note "2022-12-13"
    根据顶部状态栏发现，有时 GPU 无监控，再运行 nvidia-smi 报错
    ```bash
    $ nvidia-smi
    Failed to initialize NVML: Driver/library version mismatch
    ```
    在想有没有方法无须重启便可重启 nvidia，看了 [:link:](https://www.reddit.com/r/linuxquestions/comments/5b7tf1/is_there_a_way_to_reload_nvidia_drivers_without/) 的讨论，最好的方式可能还是直接重启。

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

The resulting status is `hi`, where the first letter implies `hold`.

```bash
~$ dpkg -l | grep "4.15.0-147"
ii  linux-headers-4.15.0-147                                    4.15.0-147.151                             all          Header files related to Linux kernel version 4.15.0
hi  linux-headers-4.15.0-147-generic                            4.15.0-147.151                             amd64        Linux kernel headers for version 4.15.0 on 64 bit x86 SMP
hi  linux-image-4.15.0-147-generic                              4.15.0-147.151                             amd64        Signed kernel image generic
hi  linux-modules-4.15.0-147-generic                            4.15.0-147.151                             amd64        Linux kernel extra modules for version 4.15.0 on 64 bit x86 SMP
hi  linux-modules-extra-4.15.0-147-generic                      4.15.0-147.151                             amd64        Linux kernel extra modules for version 4.15.0 on 64 bit x86 SMP
```

!!! note "2022-06-13"
    However, currently the kernel is

    ```bash
    ~$ date
    Mon 13 Jun 2022 04:17:43 PM CST
    ~$ uname -a
    Linux weiya-ThinkPad-T460p 5.4.0-110-generic #124-Ubuntu SMP Thu Apr 14 19:46:19 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
    ```

    so it is outdated to hold such kernel. To undo such operation, 

    ```bash
    $ sudo apt-mark unhold 4.15.0-147-generic
    ```

    Then it will be removed if we perform autoclean

    ```bash
    ~$ sudo apt autoremove --purge 
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following packages will be REMOVED:
    linux-headers-4.15.0-147* linux-headers-4.15.0-147-generic* linux-image-4.15.0-147-generic* linux-modules-4.15.0-147-generic* linux-modules-extra-4.15.0-147-generic*
    ```

    Note that this command will still keep one old kernel, along with the currently running one [:link:](https://linuxconfig.org/how-to-remove-old-kernels-on-ubuntu). For example,

    ```bash
    ~$ dpkg -l | grep "linux-image" | grep '^ii'
    ii  linux-image-5.4.0-110-generic                               5.4.0-110.124                              amd64        Signed kernel image generic
    ii  linux-image-5.4.0-117-generic                               5.4.0-117.132                              amd64        Signed kernel image generic
    ii  linux-image-generic                                         5.4.0.117.120                              amd64        Generic Linux kernel image
    ```

    However, here is a warning,

    !!! warning
        dpkg: warning: while removing linux-modules-4.15.0-147-generic, directory '/lib/modules/4.15.0-147-generic' not empty so not removed
