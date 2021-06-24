# Disqus for mainland China

如果能够正常访问 Disqus，直接参考官方文档进行配置，或者这篇博客，[Adding Disqus to a Jekyll Blog](https://sgeos.github.io/jekyll/disqus/2016/02/14/adding-disqus-to-a-jekyll-blog.html)。

目前使用 [fooleap/disqus-php-api](https://github.com/fooleap/disqus-php-api) 的方案，详见

- [科学使用 Disqus](http://blog.fooleap.org/use-disqus-correctly.html)
- [检测网络是否能够访问 Disqus](http://blog.fooleap.org/check-network-able-to-access-disqus.html)

另外，[博客统计报告（2016 上半年）](https://imququ.com/post/first-half-of-2016-blog-analytics.html#simple_thread)
的评论系统也是作者使用 Disqus API 写的，不过目前似乎还没有剥离出来

> @Jerry Qu: 不是不想开源啊，主要是这部分逻辑跟我博客系统从 Node.js 到 JS 到 CSS 全部耦合在一起了。

## disable ads

use the following jquery scripts,

```bash
<script src="/js/jquery-latest.min.js"></script>
<!--rm disqus ad refer to https://www.javaer101.com/article/25891160.html-->
<script>
  (function($){
    setInterval(() => {
        $.each($('iframe'), (arr,x) => {
            let src = $(x).attr('src');
            if (src && src.match(/(ads-iframe)|(disqusads)/gi)) {
                $(x).remove();
            }
        });
    }, 300);
})(jQuery);
```

as used in my [Chinese blog](https://blog.hohoweiya.xyz/2021/05/02/back/).

## Reconfigure on AWS 

之前是在 Vultr 上买了服务器，但是花费也不小，遂改用 AWS free tier.

### Step 1: Install Apache2

```bash
sudo apt update
sudo apt install apache2
```

配置云服务商的安全组，添加 80,443 端口，则应该能够访问 apache 的默认网页。

然后在 `/var/www` 下新建 `disqus` 文件夹

```bash
sudo mkdir /var/www/disqus
sudo chown -R $USER:$USER /var/www/disqus
```

并配置

```bash
sudo nano /etc/apache2/sites-available/disqus.conf
```

可以直接复制 `000-default.conf`，然后更改 `ServerName` 为域名，而 `DocumentRoot`改成 `/var/www/disqus`.

然后启动 `disqus.conf` 并关闭 `000-default.conf`

```bash
sudo a2ensite disqus.conf
sudo a2dissite 000-default.conf
```

这样的好处是以后方便添加不同的 hosts。

### Step 2: Install php

直接下载就好了

```bash
sudo apt install php
```

默认装的是 php7.2，虽然文档中说的是 php5.6，切换不同版本的 php 可以参考 [How to install php5 and php7 on Ubuntu 18.04 LTS](https://vitux.com/how-to-install-php5-and-php7-on-ubuntu-18-04-lts/)

主要有以下两种方式，

- 通过 Apache2, 即

```bash
# disable
sudo a2dismod php5.6
sudo a2enmod php7.2
```

- 通过 `update-alternatives`

```bash
sudo update-alternatives --set php /usr/bin/php7.2
# or select from the list
sudo update-alternatives --config php
```


另外还要安装 `cURL`，不然会报错

> PHP Fatal error:  Call to undefined function curl_init() in /var/www/disqus/disqus-php-api/api/init.php on line 104

其中`curl_init()` 是 cURL 的函数。需要注意选择与 php 匹配的版本

```bash
sudo apt-get install php7.2-curl
```

### Step 3: Install api


```bash
cd /var/www/disqus
git clone git@github.com:fooleap/disqus-php-api.git
```

然后配置 `config.php`，似乎也要给 api 文件夹写的权限，不然会报出

![](disqus-api.png)

然后在 [Disqus Api](https://disqus.com/api/applications/) 中更新 callback url，同时也要更新下 CNAME 记录。

最后可以在浏览器中访问 `****/login.php` 看是否配置成功，如果不成功，可以查看日志文件寻找 bug，

```bash
cat /var/log/error.log
```

有一点比较困惑的是，disqus 会报出，

![](disqus-api-1.png)

于是我手动把 `login.php` 文件中的 `$redirect` 改成 callback url，这样确实也成功了，但是我肯定错过了什么东西，也许 Apache 目录名的设置？

后来通过在 `login.php` 文件中添加打印输出语句

```bash
file_put_contents('php://stderr', print_r($redirect, TRUE));
```

似乎也只是 `http` 和 `https` 的区别，此时已经在 apache 配置文件指定了 host name 为域名。于是可以用 [`str_ireplace`](https://stackoverflow.com/questions/5289272/php-replace-http-with-https-in-url) 来完成替换。

另外，设置不允许查看 Apache 的目录，参考 [How to make Apache more secure by hiding directory folders](https://www.techrepublic.com/article/how-to-make-apache-more-secure-by-hiding-directory-folders/)

```bash
$ sudo vi /etc/apache2/apache2.conf
```
将
```bash
<Directory /var/www/>
​     Options Indexes FollowSymLinks
​     AllowOverride None
​     Require all granted
​</Directory>
```

改成

```bash
<Directory /var/www/>
​     Options FollowSymLinks
​     AllowOverride None
​     Require all granted
​</Directory>
```

然后

```bash
sudo service apache2 restart
```

## 20201001: AWS -> WSL

因为 aws 的 free tier 到期了，所以需要换一个境外服务器来转发 disqus 的评论，想到可以直接用办公室的电脑。

Step 1 与上文完全一致，只不过不需要配置安全组，只不过在本地试图访问 `localhost`，如果跳出网络配置的对话框，点击允许就好。

Step 2 直接按照 php 就好，因为并没有 php5.6，即

```bash
sudo apt install php
sudo apt-get install php7.2-curl
```

在 WSL 中启动服务器时，出现

```bash
protocol not available ah00076 failed to enable apr_tcp_defer_accept
```

这个在 [APR_TCP_DEFER_ACCEPT error when starting Apache2 #1953](https://github.com/microsoft/WSL/issues/1953) 有讨论，解决方案是将 WSL1 升级为 WSL2

在 Windows command prompt 中查看 WSL 的版本

```bash
C:\> wsl -l -v
```

升级命令为

```bash
C:\> wsl --set-version Ubuntu 2
```

其中 `Ubuntu` 可能还需要带上具体版本，以上一个命令访问的结果为准。然后会有升级提示，但是后来发现即便下载了升级包，仍然需要在 BIOS 中进行设定，而现在远程不方便重启。只好作罢。

后来再次试图 start apache，发现这这只是个 warning，并不影响运行。


Step 3 不需要重新下载了，直接把 aws 服务器中的文件夹复制过来就好。

然后运行

```bash
./ngrok http 80
```

即可通过外网访问该服务器，因为域名已经解析至 CDN，所以直接在 CDN 配置中更改回源域名，不过此时回源 Host 的配置需要格外注意。

如果回源 Host 仍设置为 `hohoweiya` 的域名，则试图通过该域名访问时，会出现 `Tunnel xxx.hohoweiya.xyz` not found，而且似乎只有当 https 才出现这样的问题。

所以回源 Host 应当设置为 ngrok 分配的域名。

其实仔细想想，也挺好理解，如果在一台已知域名的服务器上使用 virtual host，配置文件中写的 `ServerName` 其实是为了通过域名来索引到该 virtual host，但是现在我真正的服务器并没有公网 ip 或者域名，只是通过 ngrok 来转发，所以如果仍将 `hohoweiya` 的域名作为回源 Host，则是试图在 ngrok 的服务器上找名为 `xxx.hohoweiya.xyz` 的 virtual host。其实隐约觉得这个也可以实现，感觉还是一个端口转发的问题，因为前面发现似乎只有 https 时不行。

这时另外一个问题出现了，现在虽然可以访问 `login.php`，但是也报出上文中出现过的 `should match predefined callback URI` 的问题，因为此时实际上是通过 ngrok 的域名访问的，所以直接的访问便是在 disqus application 后台更改 callback URI。

## 20210518: WSL -> docker

因为今天办公室临时停电，为了不影响评论系统，于是将其短暂迁移至本地。虽然直接在本地装 apache2 + php 也是可行的，但是为了避免潜在对其它程序造成的影响，遂决定使用 docker。

一开始使用 `alexcheng/apache2-php7`。步骤如下：

### Step 0: 提前备份

提前从 WSL 中备份出 `/var/www/disqus-php-api` 文件夹至本地 `/media/weiya/PSSD/disqus20210518/`，注意仍保留文件夹 `disqus-php-api`

### Step 1: 启动 docker

```bash
$ docker run --name apache2php7 -p 10080:80 -v /media/weiya/PSSD/disqus20210518/:/var/www/disqus/ alexcheng/apache2-php7
```

此时运行日志文件直接输出到了终端

### Step 2: 进入 docker 进行配置

```bash
$ docker exec -it apache2php7 bash
```

修改 `/etc/apache2/apache2.conf` 配置文件，将 

```bash
DocumentRoot /var/www/html
```

改至

```bash
DocumentRoot /var/www/disqus
```

此时应该能够在本地浏览器中访问 `http://127.0.0.1:10080/disqus-php-api/api/login.php`，当然忽略 `should match predefined callback URI` 的问题。

### Step 3: 外网访问

首先下载 ngrok，然后启动并运行 `./ngrok http 10080`。

然后在阿里云 CDN 管理平台上设置修改源站信息及回源 Host 为 ngrok 分配的域名。

此时评论系统无论在内地还是境外都可以正常加载。

### Step 4: 测试评论

评论可以正常发出，但是没有收到邮件，一度怀疑是 php 发送邮件的模块出问题了，而且因为此 docker image 的特殊设置，

```bash
$ cat /etc/apache2/apache2.conf
...
ErrorLog /proc/self/fd/2
...
CustomLog /proc/self/fd/1 combined
```

其中 `/proc/self/fd/2` 类似 `>&2`，详见 [What's the difference between “>&1” and “>/proc/self/fd/1” redirection?](https://unix.stackexchange.com/questions/295883/whats-the-difference-between-1-and-proc-self-fd-1-redirection)，即便修改为

```bash
ErrorLog ${APACHE_LOG_DIR}/error.log
```

被记录的 error log 也只是下面这种无关痛痒的记录，

```bash
[Tue May 18 08:46:56.532391 2021] [mpm_prefork:notice] [pid 7777] AH00163: Apache/2.4.18 (Ubuntu) PHP/7.1.11 configured -- resuming normal operations 
[Tue May 18 08:46:56.532422 2021] [core:notice] [pid 7777] AH00094: Command line: '/usr/sbin/apache2'
```

详细解释见 [Apache is OK, but what is this in error.log - [mpm_prefork:notice]?](https://serverfault.com/questions/607873/apache-is-ok-but-what-is-this-in-error-log-mpm-preforknotice)

没有头绪，索性重新换了 docker image，[nimmis/apache-php7](https://hub.docker.com/r/nimmis/apache-php7)，然后重新走一遍上述步骤，这个完全就是子系统的 apache2，没发现有特意更高的设置，所以更加熟悉，比如在禁止目录访问时，需要修改

```diff
<Directory /var/www/>
-​     Options Indexes FollowSymLinks
+​     Options FollowSymLinks
​     AllowOverride None
​     Require all granted
​</Directory>
```

而且此时 `error.log` 可以记录 php 中的输出了。后来就在 php 文件中添加输出语句

```bash
file_put_contents('php://stderr', print_r($_POST, TRUE));
```

进行 debug。不过后来才发现邮件系统本身仍是正常的，只是因为此前一开始都是在 disqus 后台进行回复，而非在具体页面上回复，前者并不通过此转发系统，故没有收到邮件。

经过这一次折腾，也熟悉了下邮件提醒机制，主要有一下几点

- 后台暂存数据形式为 `md5(name+email): real_email`，注意此处 `email` 实际上是从 Disqus 请求返回值，即类似 `s****@gmail.com` 的形式，并不是全明文 `real_email`
- 父评论返回时间戳 `time` 为其 code，而子评论返回父评论的 `md5(name+email)`，因而可以找到暂存的邮箱进行提醒

### login?

另外，`login.php` 似乎用不上，所以 callback url 似乎无需修改。之前是直接用 ngrok 访问的域名，试图换成 `api.hohoweiya.xyz` 后，仍然报出 Invalid Request: should match predefined callback URI。但是换个角度，登录在内地本身就不可行，因为最终还是要 call back 到 disqus（内地无法访问）。如果用户已经翻墙，则直接呈现的是 disqus 原生窗口，登录也不在话下。所以此转发系统主要服务于匿名（指未登录，但仍记录了名字与邮箱）评论。

## 20200520: docker -> WSL

办公室供电恢复正常了，遂考虑将服务系统切回。因为这两天并没有新评论，所以无需更新暂存邮箱数据。于是只需两步

1. 开启 `./ngrok http 80` 并在阿里云后台修改域名回源
2. 开启 apache2 服务
