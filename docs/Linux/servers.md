# Servers

![](https://user-images.githubusercontent.com/13688320/126885780-549bf05b-b4ba-4882-a097-bf8778933848.png)

## SSH

### initial start

首先在本地新建 ssh key，

```bash
ssh-keygen -t [rsa | ed25519 | ecdsa | dsa]
```

!!! tip "ssh 常见 key 格式"
	参考 [更新SSH key为Ed25519](https://neil-wu.github.io/2020/04/04/2020-04-04-SSH-key/)

	- DSA: 不安全
	- RSA: 安全性依赖于key的大小，3072位或4096位的key是安全的，小于此大小的key可能需要升级一下，1024位的key已经被认为不安全。
	- ECDSA:  安全性取决于你的计算机生成随机数的能力，该随机数将用于创建签名，ECDSA使用的NIST曲线也存在可信赖性问题。
	- Ed25519: 目前最推荐的公钥算法

然后会在本地生成 `~/.ssh` 文件夹。

- 秘钥(`~/.ssh/id_rsa`): sensitive and important!!
- 公钥(`~/.ssh/id_rsa.pub`): contains the public key for authentication.  These files are not sensitive and can (but need not) be readable by anyone.
- 公钥授权文件(`~/.ssh/authorized_keys`)

将登录端的 `id_rsa.pub` 内容复制到服务器端的 `authorized_keys` 文件中即可。除了手动复制，也可以通过命令行，如

```bash
$ ssh-copy-id -p 30013 weiya@127.0.0.1
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
weiya@127.0.0.1's password: 

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh -p '30013' 'weiya@127.0.0.1'"
and check to make sure that only the key(s) you wanted were added.
```

正确输入登录密码后，便复制成功了，注意此时并未登录至目标服务器。

对于 AWS，登录需要使用 `.pem` 文件，即

```bash
$ ssh -i YourKey.pem user@host
```

而复制文件则为

```bash
$ scp -i YourKey.pem YourFile user@host:~/YourFile
```

### two consecutive ssh

```bash
$ ssh -t user@A ssh user@B
```

where `-t` avoid the warning that 

> Pseudo-terminal will not be allocated because stdin is not a terminal.

which would freeze the session.

If the usernames are the same, the second username can be omitted.

The port forwarding would be more clear. For example, open a jupyter session on node `B`, whose login node is `A`, then access the jupyter in the local browser `http://127.0.0.1:28888` after running

```bash
$ ssh -t -L 28888:localhost:8888 user@A ssh -L 8888:localhost:8888 user@B
```

### ssh until succeed

```bash
$ until ./login_lab.sh; do sleep 5; done
```

refer to [How to run ssh command until succeeded?](https://unix.stackexchange.com/questions/404792/how-to-run-ssh-command-until-succeeded)

### config file

Although it would be convenient to write a simple script `login_xxx.sh` to avoid to type the account and hostname, it would be annoying when using `scp`. It is still possible to define custom functions such as `scp_to_xxx` or `scp_from_xxx`, but too many functions might be confusing and forget the detailed definitions. 

Maybe we can try to write a config file (refer to [Configuring your favourite hosts in SSH](https://mattryall.net/blog/ssh-favourite-hosts)), in which we can define an alias for a hostname, and also specify the username, e.g., after defining

```bash
Host XX
Hostname REAL.HOSTNAME
User weiya
```

then I can just type `ssh XX` to access this server, and `scp` would also be much simpler, `scp file XX:~/`. More importantly, we can use tab-complete when entering the path, which cannot be enabled by custom functions `scp_to_xx`. 

!!! warning
	On the rocky server, it throws,
	```bash
	Bad owner or permissions on ~/.ssh/config
	```
	although the personal PC works well with same permission `-rw-rw-r--`. Refer to [ssh returns “Bad owner or permissions on ~/.ssh/config”](https://serverfault.com/questions/253313/ssh-returns-bad-owner-or-permissions-on-ssh-config), change the permission
	```bash
	$ chmod 600 ~/.ssh/config
	```
	that is, `-rw-------`.

### run GUI remotely/locally

```bash
weiya@T460p:~$ ssh weiya@G40
weiya@G40:~$ export DISPLAY=:0
weiya@G40:~$ firefox
```

如果不通过第二行来设置 DISPLAY，则会报错，

> Error: no DISPLAY environment variable specified


另外 `:0` 可以通过在服务器端运行

```bash
weiya@G40:~$ w
 20:29:30 up 10:01,  2 users,  load average: 1.53, 1.42, 1.40
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
weiya    :0       :0               10:28   ?xdm?  22:24   0.00s /usr/lib/gdm3/gdm-x-session --run-script env GNOME
```

进行查看，其中 `FROM` 栏下的 `:0` 即为当前 display 号码。

参考 [How to start a GUI software on a remote Linux PC via SSH](https://askubuntu.com/questions/47642/how-to-start-a-gui-software-on-a-remote-linux-pc-via-ssh)

如果想要在本地运行服务器端的 GUI 程序，即将服务器端的窗口发送到本地，则登录时需要加上 `-X` 选项，

```bash
ssh -X
```

To speed up the GUI loading if necessary, try to enable compression, `-C`, refer to [Why is Firefox so slow over SSH? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/187415/why-is-firefox-so-slow-over-ssh)

### `scp`

- `scp` a file with name including colon

add `./` before the file, since it will interpret colon `x:` as `[user@]host prefix` even if the filename has been wrapped with `"`.

refer to [How can I scp a file with a colon in the file name?](https://stackoverflow.com/questions/14718720/how-can-i-scp-a-file-with-a-colon-in-the-file-name)

### SeverAliveInterval and ClientAliveInterval

- `SeverAliveInterval` and `SeverAliveCountMax` are set on the client side, i.e., `~/.ssh/config`
- `ClientAliveInterval` and `ClientAliveCountMax` are set on the server side, i.e., `/etc/ssh/sshd_config`

refer to [What do options `ServerAliveInterval` and `ClientAliveInterval` in sshd_config do exactly?](https://unix.stackexchange.com/questions/3026/what-do-options-serveraliveinterval-and-clientaliveinterval-in-sshd-config-d)

## 安装 spark

~~在内地云主机上，[官网下载地址](https://spark.apache.org/downloads.html) 还没 5 秒就中断了，然后找到了[清华的镜像](https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-2.4.4/)~~

第二天发现，其实不是中断了，而是下载完成了，因为那个还不是下载链接，点进去才有推荐的下载链接，而这些链接也是推荐的速度快的镜像。

顺带学习了 `wget` 重新下载 `-c` 和重复尝试 `-t 0` 的选项。


upgrade Java 7 to Java 8:

最近 oracle 更改了 license，导致 [ppa 都用不了了](https://launchpad.net/~webupd8team/+archive/ubuntu/java)

[源码安装](https://www.vultr.com/docs/how-to-manually-install-java-8-on-ubuntu-16-04)

而且第一次听说 [`update-alternatives`](https://askubuntu.com/questions/233190/what-exactly-does-update-alternatives-do) 命令，有点类似更改默认程序的感觉。

接着按照 [official documentation](https://spark.apache.org/docs/latest/) 进行学习

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

## 重装nginx

想重装nginx，把/etc/nginx也一并删除了，但是重新安装却报错找不到conf文件。

参考[How to reinstall nginx if I deleted /etc/nginx folder (Ubuntu 14.04)?](https://stackoverflow.com/questions/28141667/how-to-reinstall-nginx-if-i-deleted-etc-nginx-folder-ubuntu-14-04)

应当用
```bash
apt-get purge nginx nginx-common nginx-full
apt-get install nginx
```

注意用 purge 不会保存配置文件，而 remove 会保存配置文件。

## CentOS 7

想直接在服务器上用 Julia 的 PGFPlotsX 画图，因为默认会弹出画好的 pdf 图象，除非按照[官方教程](https://kristofferc.github.io/PGFPlotsX.jl/v0.2/man/save.html#REPL-1)中的设置

```julia
PGFPlotsX.enable_interactive(false)
```

本来期望着用 evince 打开，但是最后竟然用 liberoffice 开开了，然后字体竟然不一致了，所以想着更改默认的 pdf 阅读软件，参考 [How to set default browser for PDF reader Evince on Linux?](https://superuser.com/questions/152202/how-to-set-default-browser-for-pdf-reader-evince-on-linux)

可以在 `.local/share/applications/mimeapps.list` 里面添加或者修改

虽然最后还是感觉通过服务器打开速度太慢了。

## Install software without root

`conda` can install many other programs, such as `tree`,

```bash
conda install -c eumetsat tree
```

The trick is to check whether the package is available via `https://anaconda.org/search?q=`

refer to [How to install packages in Linux (CentOS) without root user with automatic dependency handling?](https://stackoverflow.com/a/52561058)

## tab fails to complete

服务器上 tab 补全失效，并且报错

> 无法为立即文档创建临时文件：设备上没有空间

因为在第三方服务器上，没有权限清理 `/tmp` 文件夹，于是参考 [解决cannot create temp file for here-document: No space left on device问题](https://blog.csdn.net/weixin_37029453/article/details/107664402)

在 `.bashrc` 中加入

```bash
export TMPDIR=$HOME/tmp
```

## Let's Encrypt

If it throws the following message,

> Client with the currently selected authenticator does not support any combination of challenges that will satisfy the CA.

it is necessary to [upgrade the Certbot.](https://community.letsencrypt.org/t/solution-client-with-the-currently-selected-authenticator-does-not-support-any-combination-of-challenges-that-will-satisfy-the-ca/49983)

## Vultr配置shadowsocks

按照之前的配置方法，不可用，于是参考[轻松在 VPS 搭建 Shadowsocks 翻墙](https://www.diycode.cc/topics/738)进行配置。

## CentOS7搭建Apache

参考资料

1. [How To Install Linux, Apache, MySQL, PHP (LAMP) stack On CentOS 7](https://www.digitalocean.com/community/tutorials/how-to-install-linux-apache-mysql-php-lamp-stack-on-centos-7)
2. [CentOS 7.2 利用yum安装配置Apache2.4多虚拟主机](http://www.linuxidc.com/Linux/2017-10/147667.htm)

按照第一个链接的指示，并不能成功访问。于是尝试参考第二个链接修改配置文件。

未果，结果按照cy的建议，释放掉了这个服务器。
