# 源码安装 RMySQL 全纪录

## 服务器环境

centos 6.5 64位 

啥都没有，还不能访问外网

## 第一次尝试

将 RMySQL 源文件复制到服务器上，

```bash
R CMD INSTALL RMySQL_0.10.14.tar.gz
```



## centos 6.5 上源码安装 mysql 5.1.73

参考

1. [Install MySQL Server 5.0 and 5.1 from source code](https://geeksww.com/tutorials/database_management_systems/mysql/installation/downloading_compiling_and_installing_mysql_server_from_source_code.php)

## centos 6.5 上源码安装 MariaDB 5.5

参考
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

