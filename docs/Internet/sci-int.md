# 访问学校内网

## [ngrok](https://ngrok.com/)

在内网服务器 Server (such as Fan's lab server, ln001 server, chpc's server)进行如下操作：

1. 下载 ngrok 并解压
2. 注册 ngrok 获得 authtoken，然后运行 `./ngrok authtoken YOUR-AUTHTOKEN` 激活账户
3. 运行 `./ngrok tcp 22 --region=jp ` 来开启 ssh 的 22 端口，其中 `-region jp` 指定服务器为日本的，默认为 `us`，亲测速度很慢。然后得到下面的运行结果

![](ngrok.png)

注意 Forwarding 一行的记录，`tcp://SOME-IP-ADDRESS:PORT`

在本地 Local Laptop 上，终端中运行

```shell
ssh -X usename-for-Server@SOME-IP-ADDRESS -p PORT
```

即可登录内网服务器。

## SSH 反向隧道

假设有一台外网服务器 OuterServer，以及内网服务器 InnerServer

- OutServer: 外网服务器，可以从本地 `Laptop` 及内网服务器 `InnerServer` 通过给定外网 IP 访问
- InnerServer: 内网服务器，无法从外网访问，但可以访问外网

=== "Laptop"
    ```bash
    ssh -L 30002:localhost:30002 OuterServer
    ```

=== "OuterServer"
    ```bash
    ssh -D 30002 -p 30001 InnerServer
    ```

=== "InnerServer"
    ```bash
    ssh -R 30001:localhost:22 Server
    ```

这样只能在 ssh 访问内网资源，可以通过 chrome 的 SwithyOmega 插件实现在浏览器中访问内网，效果和登录 VPN 一样，可以正常下文献。

在 SwitchyOmega 中添加 socks5://127.0.0.1:30002 即可访问内网资源。

另外，如果只想科学上网，直接

```bash
# Local Laptop
ssh -D 30002 my@server
```

然后在 SwitchyOmega 中添加 socks5://127.0.0.1:30002。

### Jupyter Notebook

当**搭建好隧道**后，可以借用该隧道使用内网服务器上的 Jupyter Notebook，

```bash
# open jupyter notebook as usual on innerserver
jupyter notebook
# return url like, 127.0.0.1:8888/?token=xxxxxxxxxxx

# on public server
ssh -L localhost:18888:localhost:8888 -p 30001 [Inner-Server's Username]@localhost

# on my local laptop
ssh -L 18888:localhost:18888 [Server's Username]@[Server's Public IP]
```

于是在本地可以通过 

```bash
127.0.0.1:18888/?token=xxxxxxxxxxx
```

访问内网的 Jupyter Notebook。

## shootback

同 SSH 反向隧道，需要自己准备一台服务器，[项目主页](https://github.com/aploium/shootback)有详细配置过程。

但是这似乎跟不加 `-g` 版本的 ssh 反向隧道一样，会存在安全性问题，相当于暴露了公网 ip. 
