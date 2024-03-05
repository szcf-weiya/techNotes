---
comments: true
---

# fail to ssh passwordlessly

!!! info
    The post can date back to [Dec 17, 2020 ](https://github.com/szcf-weiya/techNotes/commit/4dada97167bae84a294d918459282e73205e842e)


像往常一样 ssh，但是报错了

```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ECDSA key sent by the remote host is
SHA256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.
Please contact your system administrator.
Add correct host key in /home/weiya/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /home/weiya/.ssh/known_hosts:42
  remove with:
  ssh-keygen -f "/home/weiya/.ssh/known_hosts" -R "chpc-login01.itsc.cuhk.edu.hk"
ECDSA host key for chpc-login01.itsc.cuhk.edu.hk has changed and you have requested strict checking.
Host key verification failed.
```

于是根据提示运行了

```bash
ssh-keygen -f "/home/weiya/.ssh/known_hosts" -R "chpc-login01.itsc.cuhk.edu.hk"
```

然后重新 ssh，但还是要求输入密码。类似的问题另见 [ssh remote host identification has changed](https://stackoverflow.com/questions/20840012/ssh-remote-host-identification-has-changed)

这其实对应了服务器上 `/etc/ssh` 文件夹下几个 pub 文件，咨询 Michael 也得到回复说最近 public fingerprint 有修改，这应该是 known hosts 的内容。

可以[以 MD5 的形式展示](https://superuser.com/questions/421997/what-is-a-ssh-key-fingerprint-and-how-is-it-generated)，

```bash
$ ssh-keygen -l -E md5 -f ssh_host_ed25519_key.pub
```

另外，[扫描 ip 或域名对应的 key](https://serverfault.com/questions/321167/add-correct-host-key-in-known-hosts-multiple-ssh-host-keys-per-hostname)

```bash
ssh-keyscan -t rsa server_ip
```

也能返回完全一致的结果，然后手动添加至 known_hosts 文件，仍然不能成功，尝试过新增其他格式的 key，

```bash
ssh-keygen -t [ed25519 | ecdsa | dsa]
```


然而统统没用。

后来跟服务器管理员反复沟通，提交 `ssh -vvv xxx &> ssh.log` 日志文件供其检查，才确认是最近服务器配置更改的原因，虽然没有明说，但是注意到 `/etc/ssh/sshd_config` 更新后不久管理员就回复说好了，问及原因，他的回答是，

>  It is related to security context which will make SELinux to block the file access. I think this required root permission to config.

!!! note "SELinux"
    SELinux是Linux内核的安全子系统，通过严格的访问控制机制增强系统安全性。一般情况下，建议开启SELinux来限制进程的权限，防止恶意程序通过提权等方式对系统进行攻击；然而，由于SELinux的严格访问控制机制，可能会导致一些应用程序或服务无法启动，因此在特定情况下（如开发、调试等），需暂时关闭SELinux。 [:link:](https://help.aliyun.com/zh/ecs/use-cases/enable-or-disable-selinux)

    

## Another Similar Issue

!!! info "Update: 2024-03-04"

    在 BI 公司内部服务器上，最近也无法通过密钥 ssh 访问，同时也无法通过浏览器访问 rstudio server 服务。后来与 IT 部门联系，确认也是 `SELinux` 的问题，解决方案是 disable SELinux,

    ```bash
    setenforce 0
    ```

    究其原因，大概是因为最近服务器有次重启，默认是开启了 SELinux，所以需要禁用。


