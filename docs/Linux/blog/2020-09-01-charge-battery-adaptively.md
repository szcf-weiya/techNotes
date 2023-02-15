---
comments: true
---

## Battery: Charge Adaptively

!!! info
    Post: [2020.09.01](https://github.com/szcf-weiya/techNotes/commit/1c24c099241918dc8000bd0d0199abb8829fe50a)
    
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
