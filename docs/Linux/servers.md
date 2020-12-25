# Servers

## Submitting Multiple Jobs Quickly

```bash tab="PBS"
/usr/bin/mpirun -v -np 28 -machinefile $PBS_NODEFILE ./myexe ${NUMBERARG} ${LETTERARG}
```

```bash tab="Bash"
#!/bin/bash
for NUMBERS in 1 2 3 4 5; do
	for LETTERS in a b c d e; do
 		 qsub -v NUMBERARG=$NUMBERS,LETTERARG=$LETTERS my_qsub_script.pbs
	done
done
```

```bash tab="Run"
chmod +x submit_multiple_jobs.sh
./submit_multiple_jobs.sh
```

refer to [Submitting Multiple Jobs Quickly](http://www.pace.gatech.edu/submitting-multiple-jobs-quickly).

## PBS passing argument list

```bash
qsub your.job -v arg1=val1,arg2=val2
```

## PBS cheat sheet

[PBS Script](PBS Script_0.pdf)

## PBSDSH

[PBSDSH](https://wikis.nyu.edu/display/NYUHPC/PBSDSH)

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

## Slurm

[学校服务器](https://www.cuhk.edu.hk/itsc/hpc/slurm.html)采用 Slurm，而系里服务器采用 PBS，常用等价格式如下：

```bash
# PBS
qsub -l nodes=2:ppn=16 -l mem=8g -N jobname -m be -M notify@cuhk.edu.hk
# Slurm
sbatch -N 2 -c 16 --mem=8g -J jobname --mail-type=[BEGIN,END,FAIL,REQUEUE,ALL] --mail-user=notify@cuhk.edu.hk
```

## 腾讯云服务器nginx failed

原因：80端口被占用
解决方法：kill掉占用80端口的

```
sudo fuser -k 80/tcp
```

重启

```
sudo /etc/init.d/nginx restart
```

## 重装nginx

想重装nginx，把/etc/nginx也一并删除了，但是重新安装却报错找不到conf文件。

参考[How to reinstall nginx if I deleted /etc/nginx folder (Ubuntu 14.04)?](https://stackoverflow.com/questions/28141667/how-to-reinstall-nginx-if-i-deleted-etc-nginx-folder-ubuntu-14-04)

应当用
```bash
apt-get purge nginx nginx-common nginx-full
apt-get install nginx
```

注意用purge不会保存配置文件，而remove会保存配置文件。

## CentOS 7

想直接在服务器上用 Julia 的 PGFPlotsX 画图，因为默认会弹出画好的 pdf 图象，除非按照[官方教程](https://kristofferc.github.io/PGFPlotsX.jl/v0.2/man/save.html#REPL-1)中的设置

```julia
PGFPlotsX.enable_interactive(false)
```

本来期望着用 evince 打开，但是最后竟然用 liberoffice 开开了，然后字体竟然不一致了，所以想着更改默认的 pdf 阅读软件，参考 [How to set default browser for PDF reader Evince on Linux?](https://superuser.com/questions/152202/how-to-set-default-browser-for-pdf-reader-evince-on-linux)

可以在 `.local/share/applications/mimeapps.list` 里面添加或者修改 

虽然最后还是感觉通过服务器打开速度太慢了。

## cluster 的 ssh 突然要密码了

像往常一样 ssh，但是报错了

```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ECDSA key sent by the remote host is
SHA256:eSpztdqzLF6rBXRWd8pCW0v4utoE5CYTUHTaUb0Qn0w.
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

然后重新 ssh，但还是要求输入密码。

这其实对应了服务器上 `/etc/ssh` 文件夹下几个 pub 文件，咨询 Michael 也得到回复说最近 public fingerprint 有修改，这应该是 known hosts 的内容。

可以[以 MD5 的形式展示](https://superuser.com/questions/421997/what-is-a-ssh-key-fingerprint-and-how-is-it-generated)，

```bash
$ ssh-keygen -l -E md5 -f ssh_host_ed25519_key.pub
```

另外 `ssh -v` 可以显示连接细节。

## disk quota in cluster

学校的 cluster disk quota 如下，

```bash
lfs quota -gh your_user_id /lustre
# 20GB by default from ITSC in /users/your_user_id
lfs quota -gh Stat /lustre
# 30TB shared by Statistics Department in /lustre/project/Stat
lfs quota -gh StatScratch /lustre
# 10TB by default from ITSC in /lustre/scratch/Stat
```

已经好几次因为 disk quota 超了使得程序崩溃，于是试图将 home 文件夹中的部分文件移动到 `/lustre/project/Stat` 中，但似乎 quota 并没有变化。

后来才发现，原来 quota 是通过 group 来控制的，其中上述命令中 `-g` 选项即表示 group，也就是说 `your_user_id`, `Stat` 和 `StatScratch` 都是 group 名字。如果文件夹是从别处移动来的，group并不会改变，只有直接在 `/lustre/project/Stat` 中新建的文件或文件夹才继承 Stat 的 group，这一点是最开始在 `Stat` 文件夹下创建自己目录 `MYFOLDER` 时通过 `chmod g+s MYFOLDER` 保证的。

于是简单的方法便是直接更改 group，

```bash
chgrp -R Stat SomeFolder
```
