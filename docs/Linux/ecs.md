# Aliyun ECS

试图为 ssh remote forward port 添加安全组。首先默认 remote forward port 仅允许通过 lo 访问，但是可以在 `/etc/ssh/sshd_config` 中加入

```bash
# https://www.ssh.com/academy/ssh/tunneling/example
GatewayPorts yes
```

来允许外来 ip 直接登录 (see als: <https://serverfault.com/questions/896784/ssh-remote-port-forwarding-gatewayports-yes-which-machine-to-specify-on>)。

自然带来一个安全性问题，所以想到添加安全组，但是添加完安全组后，发现安全组外的 ip 仍能正常访问。一度怀疑 remote forward port 不受安全组的控制，但又转念一想不太可能啊。

最终发现自己最初的一条安全组记录为，

![image](https://user-images.githubusercontent.com/13688320/128313509-9ae6ae36-1c6c-4aeb-b311-ae2be54e7d6b.png)

竟然允许任意地址从任意端口进行访问，而且这条记录的古怪之处在于优先级竟然为 110，但值应该介于 1 与 100 间，且越低优先级越高，而且此记录“修改”按钮不可用。

随后便删了这条规则，然后立马不能访问了。于是再添加 22 端口的规则，此后继续测试发现 remote forwarding 的端口也能使用了。