# Servers

## 建立 ssh 的信任关系

首先在本地新建 ssh key，

```bash
ssh-keygen -t [rsa | ed25519 | ecdsa | dsa]
```

!!! tip "ssh 常见 key 格式"
	参考 [更新SSH key为Ed25519](https://neil-wu.github.io/2020/04/04/2020-04-04-SSH-key/)

	- DSA: 不安全
	- RSA: 安全性依赖于key的大小，3072位或4096位的key是安全的，小于此大小的key可能需要升级一下，1024位的key已经被认为不安全。
	- ECDSA:  安全性取决于你的计算机生成随机数的能力，该随机数将用于创建签名，ECDSA使用的NIST曲线也存在可信赖性问题。
	- Ed25519: 目前最推荐的公钥算法

然后会在本地生成 `~/.ssh` 文件夹。

- 秘钥(`~/.ssh/id_rsa`): sensitive and important!!
- 公钥(`~/.ssh/id_rsa.pub`): contains the public key for authentication.  These files are not sensitive and can (but need not) be readable by anyone.
- 公钥授权文件(`~/.ssh/authorized_keys`)

将登录端的 `id_rsa.pub` 内容复制到服务器端的 `authorized_keys` 文件中即可。

## 远程运行服务器端的gui程序

```bash
weiya@T460p:~$ ssh weiya@G40
weiya@G40:~$ export DISPLAY=:0
weiya@G40:~$ firefox
```

如果不通过第二行来设置 DISPLAY，则会报错，

> Error: no DISPLAY environment variable specified


另外 `:0` 可以通过在服务器端运行

```bash
weiya@G40:~$ w
 20:29:30 up 10:01,  2 users,  load average: 1.53, 1.42, 1.40
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
weiya    :0       :0               10:28   ?xdm?  22:24   0.00s /usr/lib/gdm3/gdm-x-session --run-script env GNOME
```

进行查看，其中 `FROM` 栏下的 `:0` 即为当前 display 号码。

参考 [How to start a GUI software on a remote Linux PC via SSH](https://askubuntu.com/questions/47642/how-to-start-a-gui-software-on-a-remote-linux-pc-via-ssh)

如果想要在本地运行服务器端的 GUI 程序，即将服务器端的窗口发送到本地，则登录时需要加上 `-X` 选项，

```bash
ssh -X
```

## Submitting Multiple Jobs

[SLURM](https://www.cuhk.edu.hk/itsc/hpc/slurm.html) and PBS are two different cluster schedulers, and the common equivalent commands are as follows:

```bash
# PBS
qsub -l nodes=2:ppn=16 -l mem=8g -N jobname -m be -M notify@cuhk.edu.hk
# Slurm
sbatch -N 2 -c 16 --mem=8g -J jobname --mail-type=[BEGIN,END,FAIL,REQUEUE,ALL] --mail-user=notify@cuhk.edu.hk
```

Sometimes, we want to submit multiple jobs quickly, or perform parallel computing by dividing a heavy task into multiple small tasks.

### PBS

Suppose there is a main program `toy.jl`, and I want to run it multiple times but with different parameters, which can be passed via the `-v` option.

=== "submit.sh"

    ``` bash
    #!/bin/bash
    for number in 1 2 3 4 5; do
    	for letter in a b c d e; do
    		qsub -v arg1=$number,arg2=$letter toy.job
    	done
    done
    ```

=== "toy.job"

	``` bash
    #!/bin/bash
	cd $HOME/PROJECT_FOLDER
	julia toy.jl ${arg1} ${arg2}
	```

=== "toy.jl"

    ``` julia
    a, b = AGRS
    println("a = $a, b = $b")
    ```

The submitting command is

```bash
$ ./submit.sh
```

and here is [an example](https://github.com/szcf-weiya/Metabolic-Network/blob/09314f0558ddccc3997bcfcecf6b7576b281c3fd/old/mcmc-with-args.job) in my private projects.

### SLURM

Suppose there is a main program `run.jl`, which runs in parallel with the number of cores `np`, and I also want to repeat the program for `N` times. To properly store the results, the index of repetition `nrep` for each job has been passed to the main program.

The arguments for job file can be passed by the `--export` option.

=== "submit.sh"

    ``` bash
    #!/bin/bash
    if [ $# == 0 ]; then
        cat <<HELP_USAGE
        $0 param1 param2
        param1 number of repetitions
        param2 node label, can be stat or chpc
        param3 number of cores
    HELP_USAGE
        exit 0
    fi
    resfolder=res_$(date -Iseconds)
    for i in $(seq 1 1 $1); do
        sbatch -N 1 -c $3 -p $2 --export=resfolder=${resfolder},nrep=${i},np=$3 toy.job
    done
    ```

=== "toy.job"

    ``` bash
    #!/bin/bash
    cd $HOME/PROJECT
    julia -p $np run.jl $nrep $resfolder
    ```

=== "run.jl"

    ``` julia
    using Distributed
    const jobs = RemoteChannel(()->Channel{Tuple}(32))
    const res = RemoteChannel(()->Channel{Tuple}(32))

    function make_jobs() end
    function do_work() end

    nrep, resfolder = ARGS
    @async make_jobs()
    ```

where

- [`HELP_USAGE`](https://stackoverflow.com/questions/687780/documenting-shell-scripts-parameters) documents shell scripts' parameters.
- `$1, $2, $3` denotes the 1st, 2nd, 3rd argument in the command line, and `$0` is the script name.

The following command runs with `N = 100, np = 4` and on the `stat` partition,

```bash
$ ./submit.sh 100 stat 4
```

which is adopted from [my private project](https://github.com/szcf-weiya/Cell-Video/blob/master/DP/parallel_oracle2.sh).

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

## exclude hosts with `-x` or `--exclude`

### `srun`

```bash
# cannot exclude
$ srun -x chpc-cn050 hostname
chpc-cn050.rc.cuhk.edu.hk
```

### `salloc`

```bash
# cannot exclude
$ salloc -x chpc-cn050 -N1
salloc: Nodes chpc-cn050 are ready for job
# cannot exclude
$ salloc -x chpc-cn050 srun hostname
salloc: Nodes chpc-cn050 are ready for job
chpc-cn050.rc.cuhk.edu.hk
# NB: exclude successfully
$ salloc -w chpc-cn050 srun -x chpc-cn050 hostname
salloc: Nodes chpc-cn050 are ready for job
srun: error: Hostlist is empty!  Can't run job.
```

### `sbatch`

```bash
# cannot exclude
$ sbatch << EOF
> #!/bin/sh
> #SBATCH -x chpc-cn050
> srun hostname
> EOF
Submitted batch job 246669
$ cat slurm-246669.out
chpc-cn050.rc.cuhk.edu.hk

# NB: exclude successfully
$ sbatch << EOF
> #!/bin/sh
> #SBATCH -w chpc-cn050
> srun -x chpc-cn050 hostname
> EOF
Submitted batch job 246682
$ cat slurm-246682.out
srun: error: Hostlist is empty!  Can't run job.
```

**Observation:**

`-x` seems not to work in the allocation step, but it can exclude nodes from the allocated nodes.

Back to the manual of `-x` option:

> Explicitly exclude certain nodes from the resources **granted** to the job.

So the exclusion seems only to perform on the granted resources instead of all nodes. If you want to allocate specified nodes, `-w` option should be used.

## job priority calculation in the cluster

the formula for job priority is given by
```bash
Job_priority =
	site_factor +
	(PriorityWeightAge) * (age_factor) +
	(PriorityWeightAssoc) * (assoc_factor) +
	(PriorityWeightFairshare) * (fair-share_factor) +
	(PriorityWeightJobSize) * (job_size_factor) +
	(PriorityWeightPartition) * (partition_factor) +
	(PriorityWeightQOS) * (QOS_factor) +
	SUM(TRES_weight_cpu * TRES_factor_cpu,
	    TRES_weight_<type> * TRES_factor_<type>,
	    ...)
	- nice_factor
```
we can find those weights
```bash
$ scontrol show config | grep ^Priority
PriorityParameters      = (null)
PrioritySiteFactorParameters = (null)
PrioritySiteFactorPlugin = (null)
PriorityDecayHalfLife   = 7-00:00:00
PriorityCalcPeriod      = 00:05:00
PriorityFavorSmall      = No
PriorityFlags           = CALCULATE_RUNNING
PriorityMaxAge          = 7-00:00:00
PriorityUsageResetPeriod = NONE
PriorityType            = priority/multifactor
PriorityWeightAge       = 0
PriorityWeightAssoc     = 0
PriorityWeightFairShare = 100000
PriorityWeightJobSize   = 0
PriorityWeightPartition = 0
PriorityWeightQOS       = 0
PriorityWeightTRES      = (null)
```
only the `PriorityWeightFairShare` is nonzero, and this agrees with
```bash
$ sprio -w
          JOBID PARTITION   PRIORITY       SITE  FAIRSHARE
        Weights                               1     100000
$ sprio -w -p stat
          JOBID PARTITION   PRIORITY       SITE  FAIRSHARE
        Weights                               1     100000
$ sprio -w -p chpc
          JOBID PARTITION   PRIORITY       SITE  FAIRSHARE
        Weights                               1     100000
```
then the formula can be simplified as
```bash
Job_priority =
	site_factor +
	(PriorityWeightFairshare) * (fair-share_factor) +
	SUM(TRES_weight_cpu * TRES_factor_cpu,
	    TRES_weight_<type> * TRES_factor_<type>,
	    ...)
	- nice_factor
```
where `TRES_weight_<type>` might be GPU, see the usage weight in the [table](https://www.cuhk.edu.hk/itsc/hpc/slurm.html), and a negative `nice_factor` can only be set by privileged users,
> Nice Factor
Users can adjust the priority of their own jobs by setting the nice value on their jobs. Like the system nice, positive values negatively impact a job's priority and negative values increase a job's priority. Only privileged users can specify a negative value. The adjustment range is +/-2147483645.

- the fairshare can be obtained via `sshare`, and the calculated priority can be obtained via `sprio`.

### references

- [Multifactor Priority Plugin](https://slurm.schedmd.com/priority_multifactor.html)
- [Slurm priorities](http://www.ceci-hpc.be/slurm_prio.html)
