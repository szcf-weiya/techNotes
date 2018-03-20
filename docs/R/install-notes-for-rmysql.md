# 源码安装 RMySQL 全纪录

## 服务器环境

centos 6.5 64位 

啥都没有，还不能访问外网

## 第一次尝试

将 RMySQL 源文件复制到服务器上，

```bash
R CMD INSTALL RMySQL_0.10.14.tar.gz
```

![](first_try.jpg)

由于服务器是 centOS，按上述提示我们需要安装 mariadb-connect-c-devel，mariadb-dev 和 mysql-dev，下面依次安装这三个依赖文件。

## mariadb-connect-c-devel

这个安装最简单，免编译，不过注意要下载对应版本。

[下载地址](https://downloads.mariadb.com/Connectors/c/connector-c-3.0.3/)

## 源码安装 mysql 5.1.73

首先找文件就找了挺久，一开始发现都是 5.7 的，而且在找 community 版的时候，并没有发现 源码包，只有打包好的 bundle，但尴尬的是服务器连上传文件的大小都有限制，140 Mb 都传不上去，只好作罢。继续寻找源码包，终于在 [Install MySQL Server 5.0 and 5.1 from source code](https://geeksww.com/tutorials/database_management_systems/mysql/installation/downloading_compiling_and_installing_mysql_server_from_source_code.php) 中找到了源码地址，并参考编译安装好 mysql。

不过需要注意的是，因为没有 sudo 权限，只能安装到自己的用户目录下，所以 `configue` 时指定安装目录。

## 源码安装 MariaDB 5.5


源码安装时需要 `cmake` 编译，而 `cmake` 服务器上也没有，所以还得源码安装一下 `cmake`，安装时注意不要选择太高的版本，不然会提示 gcc 版本过低，所以选择与 gcc 适配的版本。

具体编译过程参考
1. [Generic Build Instructions](https://mariadb.com/kb/en/library/generic-build-instructions/)


### step 1

```bash
mkdir build-mariadb
cd build-mariadb
```

### step 2

注意设置 prefix

参考 [What is cmake equivalent of 'configure --prefix=DIR && make all install '?](https://stackoverflow.com/questions/6003374/what-is-cmake-equivalent-of-configure-prefix-dir-make-all-install)

```bash
cmake ../mariadb-5.*.* -DCMAKE_INSTALL_PREFIX:PATH=~/mariadb
```

### step 3

```bash
make 
make install
```

## 第二次尝试

似乎都装好了，不过测试之前，得先设置一下环境变量。再次尝试，编译时稍微前进了一些，但还是夭折了，错误如下

![](second_try.jpeg)

试着理解一下，找不到libmysqlclient.so.18，所以如果本机上有这个文件，则将其复制到 R 的library中即可。

利用 `locate` 发现虽然没有 `libmysqlclieng.so.18` 文件，但有一个 `libmysqlclieng.so.16` 文件，猜测这两个是一样的，只要将后者复制为前者，然后将其添加到`LD_LIBRARY_PATH`中，就好了。

再次安装，成功，完美！



