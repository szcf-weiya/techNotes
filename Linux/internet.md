# Environment
1. ubuntu 16.04
2. XX-net
3. privoxy

# Install

- 安装privoxy

```bash
sudo apt-get install privoxy
```

- 安装XX-net

直接访问项目主页[XX-net](https://github.com/XX-net/XX-Net)，内有安装说明

- 设置privoxy
在/etc/privoxy/config文件中加入

```bash
forward / 127.0.0.1:8087
listen-address 0.0.0.0:8118
```

# Usages

- 可以在访问时设置代理

```bash
curl ipinfo.io/ip --proxy 127.0.0.1:8087
```

- 也可以在.bashrc设置，这样所有的终端的网络访问都会走终端，在~/.bashrc文件中加入

```bash
export https_proxy=127.0.0.1:8087
export http_proxy=127.0.0.1:8087
```

然后

```bash
source ~/.bashrc
curl ipinfo.io/ip #测试是否代理成功
```

# Notes

3.10 15:36 chrome 在升级xxnet后成功翻墙了，而Firefox不行了。未升级前情况是相反的，但未升级时Firefox从不能用到能用。

# 有线

```bash
sudo su
ifconfig enp0s31f6 up
ifconfig enp0s31f6 10.71.115.59 netmask 255.255.255.0 up
ping 10.71.115.254
route add default gw 10.71.115.254 enp0s31f6
ping 10.71.45.100
```
