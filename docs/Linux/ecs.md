---
comments: true
---

# AliCloud ECS

## 安全组设置

试图为 ssh remote forward port 添加安全组。首先默认 remote forward port 仅允许通过 lo 访问，但是可以在 `/etc/ssh/sshd_config` 中加入

```bash
# https://www.ssh.com/academy/ssh/tunneling/example
GatewayPorts yes
```

来允许外来 ip 直接登录 (see also: <https://serverfault.com/questions/896784/ssh-remote-port-forwarding-gatewayports-yes-which-machine-to-specify-on>)。

自然带来一个安全性问题，所以想到添加安全组，但是添加完安全组后，发现安全组外的 ip 仍能正常访问。一度怀疑 remote forward port 不受安全组的控制，但又转念一想不太可能啊。

最终发现自己最初的一条安全组记录为，

![image](https://user-images.githubusercontent.com/13688320/128313509-9ae6ae36-1c6c-4aeb-b311-ae2be54e7d6b.png)

竟然允许任意地址从任意端口进行访问，而且这条记录的古怪之处在于优先级竟然为 110，但值应该介于 1 与 100 间，且越低优先级越高，而且此记录“修改”按钮不可用。

随后便删了这条规则，然后立马不能访问了。于是再添加 22 端口的规则，此后继续测试发现 remote forwarding 的端口也能使用了。

## 子网掩码

想在云服务器后台设置安全组规则，只允许 CUHK 的网进入；但是填写时，除了 IP，还需要子网掩码，经查为 [CUHKNET](https://ipinfo.io/AS3661/137.189.0.0/16)

> 137.189.0.0/16

一直没懂子网掩码是啥，比如经常看到的 `255.255.255.0`，但是这里又有个 `/16`，所以很懵。[这篇知乎回答](https://www.zhihu.com/question/56895036)解决了我的疑问。简单说，子网掩码也是 4 组长度为 8 的二进制数组成，从左到右前 `n` 位为 1，其余为 0，此时子网掩码记为 `/n`，或者换成十进制数。

它可用于判断两个 ip 是否属于同一网段，以及该网段有多少个 ip。另外可以得到网络号（ip中对应掩码为1的部分）和主机号（ip中对应掩码为0的部分）。

反过来，给定一个 IP，想确定出其所在的网段。可以按如下步骤，以添加 Harvard University WiFi 为例，

1. 访问 <https://ipinfo.io/ip> 确定 ip，如 `67.134.206.47`
2. 通过 ASN API <https://ipinfo.io/products/asn-api> 输入上述 IP，得到 ASN 编号，`AS1742`
3. 访问 <https://ipinfo.io/AS1742> 获取全部 ip address range, 其中有一条便是，`67.134.204.0/22`

下面验证 `67.134.206.47` 是否属于 `67.134.204.0/22`

```julia
julia> bitstring(204)[end-7:end]
"11001100"

# only consider the 3rd field, 22 - 8 - 8 = 6
julia> bitstring(204)[end-7:end][1:6]
"110011"
```

那么第三位允许的便有四种可能，

```julia
julia> parse(Int, "11001100", base = 2)
204

julia> parse(Int, "11001101", base = 2)
205

julia> parse(Int, "11001110", base = 2)
206

julia> parse(Int, "11001111", base = 2)
207
```

