# Servers

## Submitting Multiple Jobs Quickly

refer to [Submitting Multiple Jobs Quickly](http://www.pace.gatech.edu/submitting-multiple-jobs-quickly).

## PBS passing argument list

```bash
qsub your.job -v arg1=val1,arg2=val2
```

## PBS cheat sheet

[PBS Script](PBS Script_0.pdf)

## 安装 spark

~~在内地云主机上，[官网下载地址](https://spark.apache.org/downloads.html) 还没 5 秒就中断了，然后找到了[清华的镜像](https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-2.4.4/)~~

第二天发现，其实不是中断了，而是下载完成了，因为那个还不是下载链接，点进去才有推荐的下载链接，而这些链接也是推荐的速度快的镜像。

顺带学习了 `wget` 重新下载 `-c` 和重复尝试 `-t 0` 的选项。


upgrade Java 7 to Java 8:

最近 oracle 更改了 license，导致 [ppa 都用不了了](https://launchpad.net/~webupd8team/+archive/ubuntu/java)

[源码安装](https://www.vultr.com/docs/how-to-manually-install-java-8-on-ubuntu-16-04)

而且第一次听说 [`update-alternatives`](https://askubuntu.com/questions/233190/what-exactly-does-update-alternatives-do) 命令，有点类似更改默认程序的感觉。

接着按照 [official documentation](https://spark.apache.org/docs/latest/) 进行学习


## AWS

1. 上传文件

```
scp -i MyKeyFile.pem FileToUpload.pdf ubuntu@ec2-123-123-123-123.compute-1.amazonaws.com:FileToUpload.pdf
```

refer to [Uploading files on Amazon EC2](https://stackoverflow.com/questions/10364950/uploading-files-on-amazon-ec2)

2. mirror 镜像

wget http://apache.mirrors.tds.net/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz

3. slave 结点连接不上 master

```
Caused by: java.io.IOException: Connecting to ×××× timed out (120000 ms)
```

安全组配置，后台允许 `7077` 端口 `In`，本来以为同在一个 VPC 不需要配置。

4. AWS 结点间免密登录

[Passwordless ssh between two AWS instances](https://markobigdata.com/2018/04/29/passwordless-ssh-between-two-aws-instances/)