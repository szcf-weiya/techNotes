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
    - dnf: Dandified Yum, RPM发行版的软件包管理器Yellowdog Updater, Modified（yum）的下一代版本
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

## shared objects `.so` (dynamic library)

### LD_PRELOAD

!!! note
    Used in [:link:]()

If it is set on the same line before the command, it is only valid on the same line. For example, 

```bash
(R4.1.0) /media/weiya/PSSD/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ ldd libR.so 
        libstdc++.so.6 => /media/weiya/PSSD/Programs/anaconda3/envs/R4.1.0/lib/R/lib/./../.././libstdc++.so.6 (0x00007f80f4d1e000)

(R4.1.0) /media/weiya/PSSD/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ldd libR.so 
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fc70eb4c000)
```

!!! warning
    it is should be a particular file instead of a folder, otherwise it throws
    ```bash
    $ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/ ldd libR.so 
    ERROR: ld.so: object '/usr/lib/x86_64-linux-gnu/' from LD_PRELOAD cannot be preloaded (cannot read file data): ignored.
    ```

In the words of [:link:](https://www.baeldung.com/linux/ld_preload-trick-what-is), 

> LD_PRELOAD is an environment variable, it affects only the current process

Alternatively, we can use `export` and `unset`, 

```bash
~/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
~/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ ldd libR.so 
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f601f456000)
~/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ conda activate R4.1.0
(R4.1.0) ~/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ echo $LD_PRELOAD 
/usr/lib/x86_64-linux-gnu/libstdc++.so.6
(R4.1.0) ~/Programs/anaconda3/envs/R4.1.0/lib/R/lib$ ldd libR.so 
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb231b15000)
```

BTW, in the above, `conda activate` does not affect.

!!! warning
    `export LD_PRELOAD` did not work in GitHub Actions, such as [:link:](https://github.com/szcf-weiya/MonotoneSplines.jl/actions/runs/3841183884/workflow).
    The proper way is to put it to `env`. [:link:](https://github.com/szcf-weiya/MonotoneSplines.jl/blob/7c2e96fb2214c6d5dc4e123ae8beb9128be14086/.github/workflows/ci.yml#L68)

!!! tip
    The documentation can be found in `man ld.so`, ~~not `man ld`~~.    

### search order

An executable look for shared objects (`.so` files) in the following locations in order

- `rpath`
- `LD_LIBRARY_PATH`
- `runpath`
- `/etc/ld.so.conf`
- `/lib`
- `/usr/lib`

refer to

- [Shared Libraries: Understanding Dynamic Loading](https://amir.rachum.com/blog/2016/09/17/shared-libraries/) **(Highly Recommended)**
- [Where do executables look for shared objects at runtime?](https://unix.stackexchange.com/questions/22926/where-do-executables-look-for-shared-objects-at-runtime)


So there are several ways to fix the NotFound error,

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

## update-alternatives

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

## Kill Processes

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

## `user` vs. `sys`

- `time` commands return three times, named `real`, `user` and `sys`, the detailed explanation refers to [What do 'real', 'user' and 'sys' mean in the output of time(1)?](https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1)
- `user space` vs `kernel space`: [维基百科](https://zh.wikipedia.org/wiki/%E4%BD%BF%E7%94%A8%E8%80%85%E7%A9%BA%E9%96%93)说，在操作系统中，虚拟内存通常会被分成用户空间（英语：User space，又译为使用者空间），与核心空间（英语：Kernel space，又译为内核空间）这两个区段。

## control android phone by PC's mouse and keyboard

[How to Control Your Android Using Your Computer’s Mouse and Keyboard](https://www.makeuseof.com/tag/control-android-using-computers-mouse-keyboard/)

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

## Run in Background 

- 命令末尾的 `&` 表示在后台运行。refer to [What does “&” at the end of a linux command mean?](https://stackoverflow.com/questions/13338870/what-does-at-the-end-of-a-linux-command-mean)

- `jobs -l` 返回后台运行程序的 `PID`，refer to [How to get PID of background process?](https://stackoverflow.com/questions/1908610/how-to-get-pid-of-background-process)

但是 `jobs` [只显示属于当前 shell 的后台程序](https://superuser.com/a/607219), 如果重新登录，则不会显示后台程序，详见 [`jobs` command doesn't show any background processes](https://superuser.com/questions/607218/jobs-command-doesnt-show-any-background-processes)

- `nohup command &` vs `command &`: 一般情况下，前者在当前 shell 关掉情况下仍能运行，后者不行；但其实如果 `shopt | grep hupon` 的选项为 off （详细解释另见 [How bash handles the jobs when logout? - Stack Overflow](https://stackoverflow.com/questions/4298741/how-bash-handles-the-jobs-when-logout)），后者也能继续运行（通过 `ps -ef | grep command` 检查运行状态），所以区别不大，不过如果有输出前者会输出到文件 `nohup.txt`。更多细节详见 [What's the difference between nohup and ampersand](https://stackoverflow.com/questions/15595374/whats-the-difference-between-nohup-and-ampersand)
    - 在 tmux session 里面，当关闭 tmux session 时，只有 `nohup ./foo.sh &` 可以继续运行，而 `./foo.sh &` 则不行，其中 `foo.sh` 中代码为 `for i in {1..100}; do echo $i; sleep 1; done`。另见我提的问题 [linux - Different behaviors of nuhup and ampersand (`&`) in tmux - Stack Overflow](https://stackoverflow.com/questions/72868743/different-behaviors-of-nuhup-and-ampersand-in-tmux)

似乎 `docker run` 使用 nohup 会直接退出，并生成文件 `nohup.txt`，里面内容为“the input device is not a TTY”.

## crontab定时任务

`* */1 * * * *` 表现为每分钟执行，但是本来第 1 列应当为分钟，而第 2 列为小时，这样使用对用法理解错误，而且改成 `* * */1 * * *` 仍然表现为每分钟。试图

```bash
sudo service cron restart
# or
sudo service cron reload
```

都失败了。所以还是理解出现了偏差，

参考[Linux 设置定时任务crontab命令](https://www.cnblogs.com/zoulongbin/p/6187238.html) 和 [关于定时执行任务：Crontab的20个例子](https://www.jianshu.com/p/d93e2b177814)

## gvim fullscreen

refer to [Is there a way to turn gvim into fullscreen mode?](https://askubuntu.com/questions/2140/is-there-a-way-to-turn-gvim-into-fullscreen-mode)

In short,

1. install wmctrl
2. map F11 via .vimrc

## `/dev/loopx`

refer to [What is /dev/loopx?](https://askubuntu.com/questions/906581/what-is-dev-loopx).

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

## different CUDA version shown by nvcc and NVIDIA-smi

refer to [Different CUDA versions shown by nvcc and NVIDIA-smi](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi)

> CUDA has 2 primary APIs, the runtime and the driver API. Both have a corresponding version
>
> - The necessary support for the driver API (e.g. libcuda.so on linux) is installed by the GPU driver installer.
> - The necessary support for the runtime API (e.g. libcudart.so on linux, and also nvcc) is installed by the CUDA toolkit installer (which may also have a GPU driver installer bundled in it).

`nvidia-smi`: installed by the GPU driver installer, and generally has the GPU driver in view, not anything installed by the CUDA toolkit installer.
`nvcc`: the CUDA compiler-driver tool that is installed with the CUDA toolkit, will always report the CUDA runtime version that it was built to recognize.

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

## You have new mail

Here is a message when I login in to the Office PC,

```bash
You have new mail.
Last login: Thu May 20 13:29:14 2021 from 127.0.0.1
```

Refer to [“You have mail” – How to Read Mail in Linux Command Line](https://devanswers.co/you-have-mail-how-to-read-mail-in-ubuntu), the message is stored in the spool file, which is located at `/var/mail/$(whoami)`. The I found that this is a failed email when I wrote the mail notification script when there are new error message in `/var/log/apache2/error.log`.

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

!!! warning
    WSL does not support `systemd`, for example, it throws

    ```bash
    $ systemctl list-units
    System has not been booted with systemd as init system (PID 1). Can't operate.
    ```

    see also
    
    - [How to make WSL run services at startup - Super User](https://superuser.com/questions/1343558/how-to-make-wsl-run-services-at-startup)

See also: [systemd/User - ArchWiki](https://wiki.archlinux.org/title/Systemd/User)

### locally via `--user`

the service can be also set up locally, such as `~/.local/share/systemd/user/ssh4lab.service` in [:link:](software.md#fail-to-open-terminal)

we can check the status and manage it via

```bash
systemctl --user status/stop/start/disable/enable ssh4lab
```

see also: [:link:](https://superuser.com/questions/1037466/how-to-start-a-systemd-service-after-user-login-and-stop-it-before-user-logout)

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

!!! tip
    连接 HDMI 线没反应，也可以用 `xrandr` 排查是 PC 的问题还是显示器的问题。如果是电脑端没有显示连接，那很可能是显示器的问题。2022-09-16 初次连接 Yale 办公室的显示器时，便是电脑端没有插好。

## System Monitor

I am currently using the [gnome-shell extension](#system-monitor) and [Netdata: Web-based Real-time performance monitoring](https://github.com/netdata/netdata)

!!! info 
    Other candidates:

    - `glances`: refer to [What system monitoring tools are available?](https://askubuntu.com/questions/293426/what-system-monitoring-tools-are-available)

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
    - `10min_cpu_iowait`: average CPU iowait time over the last 10 minutes 
- `ipv4.udperrors`
    - `1m_ipv4_udp_receive_buffer_errors`： average number of UDP receive buffer errors over the last minute 
- `disk_space_usage`: disk / space utilization
- `linux_power_supply_capacity`: percentage of remaining power supply capacity
- `10s_ipv4_tcp_resets_received`: average number of received TCP RESETS over the last 10 seconds. This can be an indication that a service this host needs has crashed. Netdata will not send a clear notification for this alarm.
- `net.enp2s0`
    - `1m_received_traffic_overflow`: average inbound utilization for the network interface enp2s0 over the last minute: check if there are attempts to attack the server via `/var/log/secure`, refer to [详解CentOS通过日志反查入侵](https://www.linuxprobe.com/centos-linux-logs.html)
- `net_fifo.enp2s0.10min_fifo_errors`: number of FIFO errors for the network interface enp2s0 in the last 10 minutes [possibly the indicator for much overflow to the disk](https://user-images.githubusercontent.com/13688320/132781160-542ca686-e797-4b29-8b8c-95a4a1849c16.png)

## Get history of other tty/pts?

seems not.

see also [How to get complete history from different tty or pts - Stack Overflow](https://stackoverflow.com/questions/51074474/how-to-get-complete-history-from-different-tty-or-pts)

