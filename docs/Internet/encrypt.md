# Encryption 

## RC4

在使用 `scp` 及 `rsync` 传输文件时，有策略指出使用弱加密算法可以提高文件传输速度，如

![image](https://user-images.githubusercontent.com/13688320/130758700-ad906dd9-6b21-493d-87eb-3885a78032cf.png)

> source: [SQL and Admin: rsync, scp, sftp speed test](http://nz2nz.blogspot.com/2018/05/rsync-scp-sftp-speed-test.html)

及 [The fastest remote directory rsync over ssh archival I can muster (40MB/s over 1gb NICs)](https://gist.github.com/KartikTalwar/4393116)

其中 `arcfour` 即是一种弱加密算法。

> 在密码学中，RC4（来自Rivest Cipher 4的缩写）是一种串流加密法，密钥长度可变。它加解密使用相同的密钥，因此也属于对称加密算法。RC4是有线等效加密（WEP）中采用的加密算法，也曾经是TLS可采用的算法之一。
>
> RC4已经成为一些协议和标准的一部分，如1997年的WEP和2003年的WPA；和1995年的SSL，以及1999年的TLS。2015年由 RFC 7465 禁止RC4在所有版本的TLS中使用。
> 
> 2015年，比利时鲁汶大学的研究人员Mathy Vanhoef及Frank Piessens，公布了针对RC4加密算法的新型攻击程式，可在75小时内取得cookie的内容。[4]
> 
> source: [Wiki: RC4](https://zh.wikipedia.org/wiki/RC4)

但是在 `stapc220` 中并不支持 `arcfour`

```bash
weiya@stapc220:~$ ssh -V
OpenSSH_7.6p1 Ubuntu-4ubuntu0.3, OpenSSL 1.0.2n  7 Dec 2017
weiya@stapc220:~$ ssh -Q cipher
3des-cbc
aes128-cbc
aes192-cbc
aes256-cbc
rijndael-cbc@lysator.liu.se
aes128-ctr
aes192-ctr
aes256-ctr
aes128-gcm@openssh.com
aes256-gcm@openssh.com
chacha20-poly1305@openssh.com
```

但是稍早一点的版本便支持 `arcfour`，

```bash
weiya@aliyun:~$ ssh -V
OpenSSH_6.6.1p1 Ubuntu-2ubuntu2.13, OpenSSL 1.0.1f 6 Jan 2014
weiya@aliyun:~$ ssh -Q cipher
3des-cbc
blowfish-cbc
cast128-cbc
arcfour
arcfour128
arcfour256
aes128-cbc
aes192-cbc
aes256-cbc
rijndael-cbc@lysator.liu.se
aes128-ctr
aes192-ctr
aes256-ctr
aes128-gcm@openssh.com
aes256-gcm@openssh.com
chacha20-poly1305@openssh.com
```

试图添加未支持的加密算法，主要思想就是在 `/etc/ssh/sshd_config` 中加入

```bash
Ciphers cipher1,cipher2,cipher3
```

参考

- [Enable arcfour and Other Fast Ciphers on Recent Versions of OpenSSH](https://mgalgs.github.io/2014/10/22/enable-arcfour-and-other-fast-ciphers-on-recent-versions-of-openssh.html)

但是最后发现似乎只能添加支持的算法（即 `ssh -Q cipher` 中的算法）

其实这个设置更适合 disable 弱加密算法，也就是说在上述配置文件中，删掉弱加密算法，这也是讨论更多的方向，如 

- [SSH: How to disable weak ciphers? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/333728/ssh-how-to-disable-weak-ciphers)
- [服务器ssh安全 - SegmentFault 思否](https://segmentfault.com/a/1190000022766288)

既然不能支持 arcfour，另一个自然的思路便是选择相对较弱的加密算法，想找出加密算法间强弱的一个比较，

![image](https://user-images.githubusercontent.com/13688320/130757199-63145ac2-ab90-4104-85d0-1dd07137383d.png)

> source: [各种加密算法比较 - 落叶的瞬间; - 博客园](https://www.cnblogs.com/sunxuchu/p/5483956.html)

但似乎各有优劣，并没有明显的强弱关系，而且也暂时没找到针对 ssh 使用的加密算法的评测。

## MD5

```bash
~$ printf "hello\n" | md5sum
b1946ac92492d2347c6235b4d2611184  -
~$ printf "hello" | md5sum
5d41402abc4b2a76b9719d911017c592  -
~$ echo -n "hello" | md5sum
5d41402abc4b2a76b9719d911017c592  -
~$ echo "hello" | md5sum
b1946ac92492d2347c6235b4d2611184  -
```

where `-n` does not output the trailing newline `\n`, but 

```bash
~$ echo -n "hello\n" | md5sum
20e2ad363e7486d9351ee2ea407e3200  -
~$ echo -n "hello\n"
hello\n~$
```

other materals releated to MD5

- [三分钟学习 MD5](https://zhuanlan.zhihu.com/p/26592209)
- [为什么现在网上有很多软件可以破解MD5，但MD5还是很流行？](https://www.zhihu.com/question/22311285/answer/20960705)
