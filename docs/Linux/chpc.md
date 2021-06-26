# Notes on CHPC-Cluster

## Login

After generating and configuring SSH key pair, we can directly access the server via

```bash
$ ssh USERNAME@chpc-login01.itsc.cuhk.edu.hk
```

since the login node is not suggested/allowed to run test jobs, it would be more convenient to login in the test node, `sandbox`. This can be done with consecutive ssh,

```bash
$ ssh -t USERNAME@chpc-login01.itsc.cuhk.edu.hk ssh sandbox
```

where `-t` aims to avoid the warning

> Pseudo-terminal will not be allocated because stdin is not a terminal.

## Custom Commands

Some of the commands would be explained in the following sections.

- aliases

```bash
# delete all jobs
alias qdelall='qstat | while read -a ADDR; do if [[ ${ADDR[0]} == +([0-9]) ]]; then qdel ${ADDR[0]}; fi ; done'
# list available cores
alias sinfostat='sinfo -o "%N %C" -p stat -N'
# list available gpu
alias sinfogpu='sinfo -O PartitionName,NodeList,Gres:25,GresUsed:25 | sed -n "1p;/gpu[^:]/p"'
# check disk quota
alias myquota='for i in `whoami` Stat StatScratch; do lfs quota -gh $i /lustre; done'
# list jobs sorted by priority
alias sacctchpc='sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocGRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn)'
# list jobs sorted by priority (only involved stat)
alias sacctstat='sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocGRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn) | sed -n "1,2p;/stat/p"'
```

- functions 

```bash
#request specified nodes in interactive mode
request_cn() { srun -p stat -q stat -w chpc-cn1$1 --pty bash -i; }
request_gpu() { srun -p stat -q stat --gres=gpu:1 -w chpc-gpu01$1 --pty bash -i; }
request_gpu_chpc() { srun -p chpc --gres=gpu:1 -w chpc-gpu$1 --pty bash -i; }
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

## Specify Nodes

The nodes can be excluded with `-x` or `--exclude`, and it can be specified with `-w`.

!!! tip "TL; DR"
    According to the following experiments, my observation is that
    > the exclusion seems only to perform on the granted resources instead of all nodes. If you want to allocate specified nodes, `-w` option should be used.

- `srun`

```bash
# cannot exclude
$ srun -x chpc-cn050 hostname
chpc-cn050.rc.cuhk.edu.hk
```

- `salloc`

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

- `sbatch`

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

## Job Priority

The submitted jobs are sorted by the calculated job priority in descending order. 

!!! tip "TL;DR"
    You can check the priority of all submitted jobs (not only yours but also others), and then you can find where you are, and figure out when your job can start to run.
    ```bash
    $ sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocGRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn)
    ```

The formula for job priority is given by

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

then the formula would be simplified as

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

refer to

- [Multifactor Priority Plugin](https://slurm.schedmd.com/priority_multifactor.html)
- [Slurm priorities](http://www.ceci-hpc.be/slurm_prio.html)

## Disk Quota

Sometimes, you might find that your job cannot continue to write out results, and you also cannot create a new file. It might imply that your quota reaches the limit, and here is a tip to "increase" your quota without cleaning your files.

!!! tip "TL;DR"
    Tip to "increase" your personal quota is to count the files as the shared department quota, so just change the group membership of your files,
    ```bash
    $ chgrp -R Stat SomeFolder
    ```    

Firstly, you can check your personal quota with

```bash
$ lfs quota -gh your_user_id /lustre
# 20GB by default from ITSC in /users/your_user_id
```

and here is two shared quota's in the whole department,

```bash
$ lfs quota -gh Stat /lustre
# 30TB shared by Statistics Department in /lustre/project/Stat
$ lfs quota -gh StatScratch /lustre
# 10TB by default from ITSC in /lustre/scratch/Stat
```

An interesting thing is that the quota is counted by the group membership of the file, so if your personal quota exceeds, you can change the group membership of some files, and then these files would count as the shared quota instead of your personal quota. To change the group membership recursively of a folder, 

```bash
$ chgrp -R Stat SomeFolder
```

A partial Chinese description,

已经好几次因为 disk quota 超了使得程序崩溃，于是试图将 home 文件夹中的部分文件移动到 `/lustre/project/Stat` 中，但似乎 quota 并没有变化。

后来才发现，原来 quota 是通过 group 来控制的，其中上述命令中 `-g` 选项即表示 group，也就是说 `your_user_id`, `Stat` 和 `StatScratch` 都是 group 名字。如果文件夹是从别处移动来的，group并不会改变，只有直接在 `/lustre/project/Stat` 中新建的文件或文件夹才继承 Stat 的 group，这一点是最开始在 `Stat` 文件夹下创建自己目录 `MYFOLDER` 时通过 `chmod g+s MYFOLDER` 保证的。

于是简单的方法便是直接更改 group，

```bash
chgrp -R Stat SomeFolder
```

如果想找出哪些文件 group 为 `sXXXX`, 可以采用

```bash
find . -group `whoami`
```

## Custom Module

The cluster manages the software versions with [module](https://linux.die.net/man/1/module), and the default module file path is

```bash
$ echo $MODULEPATH
/usr/share/Modules/modulefiles:/etc/modulefiles
```

which requires `sudo` privilege. A natural question is whether we can create custom (local) module file to switch the software which are installed by ourselves or have not been added into the modules.

Here is an example. There is an installed R in `/opt/share/R` named `3.6.3-v2`, which does not have the modulefile, since the same version `3.6.3` is already used. But there are still differences between these two "same" versions, `3.6.3-v2` supports figures, such as "jpeg", "png", "tiff" and "cairo", while `3.6.3` not,

```bash
(3.6.3) > capabilities()
       jpeg         png        tiff       tcltk         X11        aqua
      FALSE       FALSE       FALSE        TRUE       FALSE       FALSE
   http/ftp     sockets      libxml        fifo      cledit       iconv
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE
        NLS     profmem       cairo         ICU long.double     libcurl
       TRUE       FALSE       FALSE        TRUE        TRUE        TRUE

(3.6.3-v2) > capabilities()
       jpeg         png        tiff       tcltk         X11        aqua
       TRUE        TRUE        TRUE        TRUE       FALSE       FALSE
   http/ftp     sockets      libxml        fifo      cledit       iconv
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE
        NLS     profmem       cairo         ICU long.double     libcurl
       TRUE       FALSE        TRUE        TRUE        TRUE        TRUE
```

To use `3.6.3-v2`, we can create our custom modulefile ([ref](https://www.sc.fsu.edu/computing/tech-docs/1177-linux-modules)),

```bash
# step 1: create a folder in your home directory
~ $ mkdir modules
# step 2: preappend your module path to the ~/.bashrc file
~ $ echo "export MODULEPATH=${MODULEPATH}:${HOME}/modules" >> ~/.bashrc
# step 3: copy the existing modulefile as a template
# here I skip the directory, just use name `R3.6`, which can also differ from the existing `R/3.6`
~ $ cp /usr/share/Modules/modulefiles/R/3.6 modules/R3.6
# step 4: modify the path to the software (`modroot`) as follows
```

=== "/usr/share/Modules/modulefiles/R/3.6"

    ```bash
    #%Module1.0#####################################################################
    ##
    proc ModulesHelp { } {
     global version modroot

    puts stderr "R/3.6.3 - sets the Environment for
             R scripts v3.6.3 (gcc verion)

    Use 'module whatis [module-info name]' for more information"
    }

    module-whatis "The R Project for Statistical Computing
    R is a free software environment for statistical computing and graphics.

    Here is the available versions:
            R/3.6.3"


    set     version         3.6.3
    set     app             R
    set     modroot         /opt/share/$app/$version

    module load pcre2

    conflict R

    setenv R_HOME $modroot/lib64/R

    prepend-path PATH $modroot/bin
    prepend-path LD_LIBRARY_PATH $modroot/lib
    prepend-path INCLUDE $modroot/include
    ```

=== "~/modules/R3.6"

    ```bash
    #%Module1.0#####################################################################
    ##
    proc ModulesHelp { } {
     global version modroot

    puts stderr "R/3.6.3 - sets the Environment for
             R scripts v3.6.3 (gcc verion)

    Use 'module whatis [module-info name]' for more information"
    }

    module-whatis "The R Project for Statistical Computing
    R is a free software environment for statistical computing and graphics.

    Here is the available versions:
            R/3.6.3"


    set     version         3.6.3
    set     app             R
    set     modroot         /opt/share/$app/${version}-v2

    module load pcre2
    module load intel
    conflict R

    setenv R_HOME $modroot/lib64/R

    prepend-path PATH $modroot/bin
    prepend-path LD_LIBRARY_PATH $modroot/lib
    prepend-path INCLUDE $modroot/include
    ```

=== "diff -u"

    ```diff
    @@ -18,10 +18,10 @@

     set     version         3.6.3
     set     app             R
    -set     modroot         /opt/share/$app/$version
    +set     modroot         /opt/share/$app/${version}-v2

     module load pcre2
    -
    +module load intel
     conflict R
    ```

Note that `module load intel` is also added, otherwise it will throws,

```bash
/opt/share/R/3.6.3-v2/lib64/R/bin/exec/R: error while loading shared libraries: libiomp5.so: cannot open shared object file: No such file or directory
```

since `libiomp5` is for `openmp` ([ref](https://stackoverflow.com/a/28254656)).

Now, you can use `3.6.3-v2` like other modules,

```bash
# load `3.6.3-v2`
$ module load R3.6
# unload `3.6.3-v2`
$ module unload R3.6
# load original `3.6.3`
$ module load R/3.6
```

## GPU Usage

Be cautious about the compatible versions between the deep learning framework (e.g., tensorflow), CUDA, and also the (CUDA) **driver** version of the node.

The driver version of the node is fixed, but fortunately it is downward compatible, i.e., a higher driver version also supports a lower CUDA version. We can check the driver version by

```bash
$ nvidia-smi 
Sat Jun 26 10:32:03 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.36.06    Driver Version: 450.36.06    CUDA Version: 11.0     |
...
```

which implies that the highest supported CUDA version is 11.0.

The available CUDA versions can be found as follows

```bash
$ module avail cuda
-------------- /usr/share/Modules/modulefiles ---------------
cuda/10.1          cuda/10.2          cuda/11.0          cuda/11.3(default) cuda/9.2
```

For the above node whose highest supported CUDA version is 11.0, the latest `cuda/11.3` would be incompatible, but others are all OK.

Now you can pick the proper tensorflow version according to the supported CUDA versions. Here is an [official configuration table](https://www.tensorflow.org/install/source#linux), which lists the compatible versions between tensorflow, python, CUDA, together with cuDNN.

Finally, you can validate if the GPU is correctly supported by running

```python
$ python
>>> import tensorflow as tf
# 1.x
>>> tf.test.is_gpu_available()
# 2.x
>>> tf.config.list_physical_devices('GPU')
```

What if you want to use the version that does not installed on the cluster, say `cuda/10.0`? We can install a local cuda and creat a [custom module](#custom-module) to call it. 

Following the instruction: [Install Cuda without root](https://stackoverflow.com/questions/39379792/install-cuda-without-root)

```bash
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
$ chmod +x cuda_10.0.130_410.48_linux
$ ./cuda_10.0.130_410.48_linux
```

then download `cuDNN`, but it requires to login. After extraction, copy the `include` and `lib` to the CUDA installation folder.

Next, create the custom module file, and finally I can use 

```bash
$ cuda load cuda10.0
```

to use the local cuda.

## Solutions to Abnormal Cases

### fail to ssh passwordlessly

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

### fail to access `~`

Z 在群里问道，他在服务器上提交 job 时，之前安装好的包不能使用。显然，很可能因为 `.libPaths()` 没有包含 `$HOME` 下的用户安装路径，但是他在登录结点上在 R 中运行 `.libPaths()`，一切正常。那么问题或许出在工作结点上，事实表明在该工作结点上采用以下任一种方式都能解决问题

- 在 R 中运行 `.libPaths("/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0")`
- 或者 `export R_LIBS_USER=/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0`

!!! note
    此处也看到一个类似的问题，但是原因不一样，在 [:material-stack-overflow:R_LIBS_USER ignored by R](https://stackoverflow.com/questions/53967385/r-libs-user-ignored-by-r) 问题中，原因是 `$HOME` 不能正常展开。
    
但是此时并不是很理解，因为按理说不同结点都是共享的。后来研究了下 [R 的启动机制](https://stat.ethz.ch/R-manual/R-devel/library/base/html/Startup.html)，

> On Unix versions of R there is also a file `R_HOME/etc/Renviron` which is read very early in the start-up processing. It contains environment variables set by R in the configure process. Values in that file can be overridden in site or user environment files: do not change `R_HOME/etc/Renviron` itself. Note that this is distinct from `R_HOME/etc/Renviron.site`.

才知道 `R_LIBS_USER` 是定义在 `Renviron` 中，

```bash
R_LIBS_USER=${R_LIBS_USER-'~/R/x86_64-pc-linux-gnu-library/4.0'}
```

其中 `${A-B}` 的语法是如果 `A` 没有设置，则令 `B` 为 `A`，注意[其与 `${A:-B}` 的区别](../shell/#default-value)。

这也难怪为什么直接在命令行中输入 `echo $R_LIBS_USER` 结果为空。

SSH 到该工作结点，发现其 prompt 并没有正确加载，直接出现 `bash-4.2$`，而一般会是 `[sXXXXX@chpc-sandbox ~]$`。

!!! note
    其实 `source .bashrc` 后能显示 `~`，但访问 `~` 仍然失败。另见 [Terminal, Prompt changed to “-Bash-4.2” and colors lost](https://unix.stackexchange.com/questions/125965/terminal-prompt-changed-to-bash-4-2-and-colors-lost)

这样一个直接后果就是无法解析用户目录 `~`，这大概也是为什么 `R_LIBS_USER` 在这个结点没有正常加载，因为上述系统配置文件 `/opt/share/R/4.0.3/lib64/R/etc/Renviron` 中使用了 `~`，于是需要用不带 `~` 的全路径。不过 `~` 其实只是指向 `/user/sXXXXX`，发现这个文件夹没有正常被连接，所以要使用 `/lustre/user/sXXXX`. 或者说是没有挂载，因为在其它结点上有以下三条挂载记录，

```bash
$ df -h
storage03:/chpc-userhome                  50T  4.4T   46T   9% /storage03/chpc-userhome
storage03:/chpc-optshare                  10T  713G  9.4T   7% /storage03/chpc-optshare
storage01:/chpc-users                     15T  6.2T  8.9T  42% /storage01/users
```

而该结点上没有。

为了验证上述想法，也手动进行 `export R_LIBS_USER=` 及 `.libPaths()`，

=== "DONE"
    ```R
    > .libPaths("/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0")
    > .libPaths()
    [1] "/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0"
    [2] "/lustre/opt_share/R/4.0.3/lib64/R/library" 
    ```

=== "DONE"
    ```R
    $ export R_LIBS_USER=/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0
    $ R
    > .libPaths()
    [1] "/lustre/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0"
    [2] "/lustre/opt_share/R/4.0.3/lib64/R/library"   
    ```

=== "FAIL"
    ```R
    > .libPaths("/users/sXXXXXXXXX/R/x86_64-pc-linux-gnu-library/4.0/")
    > .libPaths()
    [1] "/lustre/opt_share/R/4.0.3/lib64/R/library"
    ```

=== "FAIL"
    ```R
    $ export R_LIBS_USER=~/R/x86_64-pc-linux-gnu-library/4.0
    $ R
    > .libPaths()
    [1] "/lustre/opt_share/R/4.0.3/lib64/R/library"
    ```

可见涉及 `~` 和 `/users/sXXXX` 的均有问题。

虽然问题已经得以解决，但是很好奇为什么访问 `~` 失败。因为根据我的理解，`/users/sXXXX` (`~`)、`/lustre/users/sXXXX` 中间应该是通过类似 soft links 形式连接的（即便可能不是，因为确实直接 `ls` 也没返回指向结果）。

后来咨询管理员才明白，他们正在将迁移用户文件夹，

![Selection_1524](https://user-images.githubusercontent.com/13688320/117969152-a3699900-b359-11eb-96b3-83e658404d7e.png)

其中黄色方块表明迁移的目标硬盘，称为 `storage`，而未框出来的地方则表明当前所在硬盘 `lustre`.

但是因为并非所有用户都已迁移完成，所以需要对这两类用户访问 `~` 的行为做不同的处理，

![](https://user-images.githubusercontent.com/13688320/117969166-a9f81080-b359-11eb-8bbe-5cf1a022f7e8.png)

- 如果用户 A 已经迁移，则其 `~` 直接指向 `/storage01/users/A`
- 如果用户 B 未迁移，则其 `~` 通过 `/storage01/users/B` （此时该路径只相当于 soft link） 指向 `/lustre/users/B`

所以无论迁移与否，访问 `~` 都需要通过上图的黄色方框。那么倘若 storage 本身挂载失败，则 `~` 解析失败，而未迁移用户正因为还未迁移，所以仍能绕过 `~` 而直接访问 `/lustre/users/sXXXX`。
