# 建立SSH的信任关系

- SSH公钥(~/.ssh/id_rsa.pub)
Contains the public key for authentication.  These files are not sensitive and can (but need not) be readable by anyone.

- 公钥授权文件(~/.ssh/authorized_keys)

Lists the public keys (DSA/ECDSA/RSA) that can be used for logging in as this user.  The format of this file is described in the sshd(8) manual page.  This file is not highly sensitive, but the recommended permissions are read/write for the user, and not accessible by others.

## 公钥和私钥

- 加密是对数据进行处理，添加保护信息，如果非法用户得到被加密过的数据，也无法获取到原始的有效数据内容，所以加密的重点在于数据的安全性。
- 认证是对数据进行处理，添加鉴权信息，如果在传输的过程中被非法篡改，接收方就会校验失败并丢弃该非法数据，所以认证的重点在于数据的合法性。

## 基于公钥和私钥的加密过程

## 基于公钥和私钥的认证过程

## 基于公钥和私钥的信任关系

将登录端的id_rsa.pub内容复制到服务器端的authorized_keys文件中。


## 远程运行服务器端的gui程序

https://askubuntu.com/questions/47642/how-to-start-a-gui-software-on-a-remote-linux-pc-via-ssh

在服务器端，运行
```
w
```
可以返回当前使用的display号码，如`:0`，于是指定display的号码
```
export DISPLAY=:0
```

## 远程运行服务器端的gui程序，并发送到客户端

```
ssh -X
```
