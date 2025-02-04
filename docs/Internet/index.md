---
comments: true
---

# 网络连接相关问题

## TCP vs UDP

- TCP/IP 是互联网相关的各类协议族的总称，比如：TCP，UDP，IP，FTP，HTTP，ICMP，SMTP 等都属于 TCP/IP 族内的协议。
- TCP/IP模型是互联网的基础，它是一系列网络协议的总称。这些协议可以划分为四层，分别为链路层、网络层、传输层和应用层。

![image](https://user-images.githubusercontent.com/13688320/120821783-59ef2100-c588-11eb-837d-5b53f261e474.png)

refer to [一文搞懂TCP与UDP的区别](https://www.cnblogs.com/fundebug/p/differences-of-tcp-and-udp.html)

## 有线连不上

手动设置ip
IPv4 settings

Address: 10.71.115.59
Netmask: 24
Gateway: 10.71.115.254
DNS server: 10.10.0.21

```bash
sudo su
ifconfig enp0s31f6 up
ifconfig enp0s31f6 10.71.115.59 netmask 255.255.255.0 up
ping 10.71.115.254
route add default gw 10.71.115.254 enp0s31f6
ping 10.71.45.100
```

see also [:link:](https://askubuntu.com/questions/995660/enp0s31f6-cable-unplugged)

### ifconfig

`ifconfig` is a command line tool for diagnosing and configuring network interfaces (software interfaces to networking hardware). 

Two types of network interfaces:

- physical: represent an actual network hardware device such as network interface controller (NIC), e.g., `eth0` represents Ethernet network card (**Note: ** the current names follow the predictable network interface naming, such as `enp0s1f6`, which consists of the physical position in the pic system, refer to [Why is my network interface named enp0s25 instead of eth0?](https://askubuntu.com/questions/704361/why-is-my-network-interface-named-enp0s25-instead-of-eth0))
- virtual:
    - loopback
    - bridges
    - VLANs
    - tunnel interfaces
    - ...

Here is the output of `ifconfig` on my T460p,

```bash
$ ifconfig
docker0: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
        inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255
        ether 02:42:22:d9:b6:b0  txqueuelen 0  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

enp0s31f6: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
        ether 50:7b:9d:bd:4a:3b  txqueuelen 1000  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device interrupt 16  memory 0xf2200000-f2220000  

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1525364  bytes 4323602450 (4.3 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1525364  bytes 4323602450 (4.3 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

wlp3s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.13.59.193  netmask 255.255.128.0  broadcast 10.13.127.255
        inet6 fe80::def:d34:a2c:da88  prefixlen 64  scopeid 0x20<link>
        ether a4:34:d9:e8:9a:bd  txqueuelen 1000  (Ethernet)
        RX packets 19688705  bytes 18661301186 (18.6 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 17848834  bytes 23268449705 (23.2 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

where

- `lo` is a special virtual network interface called **loopback device**. Loopback is used mainly for diagnostics and trobuleshooting, and to connect to services running on local host
- `docker0` is a virtual bridge interface created by Docker. This bridge creates a separate network for docker containers and allows them to communicate with each other.
- flag
    - `UP`: kernel modules related to the interface have been loaded and interface is activated.
    - `BROADCAST`: interface is configured to handle broadcast packets, which is required for obtaining IP address via DHCP
    - `RUNNING`: the interface is ready to accept data
    - `MULTICAST`: the interface supports multicasting
- `MTU`: maximum transmission unit. see also [WIKI](https://en.wikipedia.org/wiki/Maximum_transmission_unit)
- `RX packets`: total number of packets received
- `TX packets`: total number of packets transmitted

Refer to [Demystifying ifconfig and network interfaces in Linux](https://goinbigdata.com/demystifying-ifconfig-and-network-interfaces-in-linux/), note that some field names are different, such as MAC address, `ether` vs `HWaddr`, see also the discussion [CentOS 7 - Networking Support: Changing ether to hwaddr](https://forums.centos.org/viewtopic.php?t=70378)

Reboot with

```bash
sudo ifconfig enp0s31f6 down
sudo ifconfig enp0s31f6 up
```

and if necessary, re-install the wifi kernel with

```bash
sudo apt-get install bcmwl-kernel-source
```

see also [:link:](https://www.shuzhiduo.com/A/D854QD6pdE/).

## Ubuntu连接UWS和eduroam

默认情况下一直连接不是，注意选对security，Authentication应当选择PEAP，如图。

![](eduroam.png)


## 阿里公共ADS差评

之前轻信了阿里公共ADS，在Ubuntu上装了，链接在[此](http://www.alidns.com/setup/#linux)，然而却无法取消设置，restore选项毫无作用。也一直不管它，在枫叶国，竟然连google都ping不通，这就很奇怪了，于是便怀疑是它搞的鬼。

为了彻底根除这个毒瘤，
```
cd /etc/resolvconf/resolv.conf.d/
sudo vim head
## 删除文件中的所有信息，其实只有阿里公共ads的配置信息
sudo vim head.save
## 删除文件中的所有信息，其实只有阿里公共ads的配置信息
```

终于好了！

## rvpn

Ubuntu下配置rvpn，需要浏览器启用JAVA插件，但新版本的chrome和firefox都不支持（firefox 52以后都不行了），解决办法是安装一个旧版本的firefox，然后按照登录rvpn时的说明进行配置就ok了。

## proxychains 实现命令行代理

参考[How to use SOCKS 5 proxy in Ubuntu command line](https://bokunokeiken.wordpress.com/2015/07/22/how-to-use-socks-5-proxy-in-ubuntu-command-line/)

```bash
proxychains curl ifconfig.me
```

!!! tip "Get Public IP via curl"
	Available websites:

	- `ifconfig.me`
	- `ipinfo.io/ip`

### `LD_PRELOAD cannot be preloaded`

在 Ubuntu 16.04 上用得好好的，但是在更新后的 Ubuntu 18.04 上使用时，报出

```bash
~$ proxychains ping www.google.com
ProxyChains-3.1 (http://proxychains.sf.net)
ERROR: ld.so: object 'libproxychains.so.3' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
PING www.google.com (199.16.156.7) 56(84) bytes of data.
```

参考 [proxychains LD_PRELOAD cannot be preloaded](https://askubuntu.com/questions/293649/proxychains-ld-preload-cannot-be-preloaded)，将 `/usr/bin/proxychains` 中的

```bash
export LD_PRELOAD=libproxychains.so.3
```

改到实际 `libproxychains.so.3` 的位置，这可以通过 locate 来确定。

另外注意到

```bash
~$ proxychains ping www.google.com
ProxyChains-3.1 (http://proxychains.sf.net)
PING www.google.com (199.59.149.244) 56(84) bytes of data.
^C
--- www.google.com ping statistics ---
2 packets transmitted, 0 received, 100% packet loss, time 1019ms
```

并不会出现类似 `curl` 时的 chain

```bash
~$ proxychains curl ipinfo.io/ip
ProxyChains-3.1 (http://proxychains.sf.net)
|DNS-request| ipinfo.io 
|S-chain|-<>-127.0.0.1:30002-<><>-4.2.2.2:53-<><>-OK
|DNS-response| ipinfo.io is 216.239.38.21
|S-chain|-<>-127.0.0.1:30002-<><>-216.239.38.21:80-<><>-OK
```

## hosts文件原理

有段时间是采用更改hosts文件来访问谷歌，但其背后的原理一直不甚清楚。突然想到这个问题，看了下面的两个博客，大致明白了。

简单来说，其目的跟dns解析域名一样，但是优先级更高，如果本机hosts文件中已经有了某域名的ip映射，则不需要通过dns返回域名ip。

更多细节可以参考这两个博客

1. [host文件的工作原理及应用](http://blog.csdn.net/tskyfree/article/details/41214829)
2. [简单科普下hosts文件原理与制作 | 老D博客](https://laod.cn/hosts/hosts-file-making.html)

## unable to resolve host

参考[http://blog.csdn.net/ichuzhen/article/details/8241847](http://blog.csdn.net/ichuzhen/article/details/8241847)

Makesure the hostname defined in `/etc/hostname` also points to `127.0.0.1` in `/etc/hosts`.

check my `/etc/hosts`

```bash
~$ cat /etc/hosts
127.0.0.1	localhost
127.0.1.1	weiya-ThinkPad-T460p
```

and also the hostname

```bash
~$ cat /etc/hostname 
weiya-ThinkPad-T460p
```

## 带宽、网速和流量

单位：

- 带宽：比特每秒 (bit/s, bps)
- 网速：字节每秒 (B/s KB/s MB/s)
- 流量：字节 (Byte)

带宽给出了理论上下载的最大速度，但实际上，

> 还有个标准（电信部门给的）：
- 512k用户的到达测速网站的速度大于 40KByte/s,即320Kbps时是属于正常的；
- 1M用户的到达测速网站的速度大于 80KByte/s,即640Kbps时是属于正常的；
- 2M以上用户的到达测速网站的速度大于 160KByte/s,即1280Kbps时是属于正常的；
- 3M以上用户的到达测速网站的速度大于 240KByte/s,即1920Kbps时是属于正常的；

参考 [浅谈带宽、网速和流量之间的关系](https://zhuanlan.zhihu.com/p/50401281)

## 反向代理

浏览博客中看到[另一种解决内地 Disqus 被墙的问题](https://blog.ichr.me/post/use-disqus-conveniently/)，采用 [DisqusJS](https://github.com/SukkaW/DisqusJS), 其中使用到了反向代理技术。

正向代理代理的对象是客户端，

![](https://pic1.zhimg.com/80/v2-07ededff1d415c1fa2db3fd89378eda0_720w.jpg?source=1940ef5c)

反向代理代理的对象是服务端

![](https://pic1.zhimg.com/80/v2-816f7595d80b7ef36bf958764a873cba_720w.jpg?source=1940ef5c)

参考 [知乎：反向代理为何叫反向代理？](https://www.zhihu.com/question/24723688)
