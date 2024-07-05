---
comments: true
---

# Notes on CHPC-Cluster @ CUHK

!!! abstract
    Here are tips for the CHPC-Cluster @ CUHK during my PhD.

    Besides the sections listed on this page, the following blogs are organized along timeline, usually recording debugging procedures,

    - [2020.12.17 Fail to SSH passwordlessly](blog/2020-12-17-passwordless-failed.md)
    - [2021.05.12 Fail to Access Home](blog/2021-05-12-fail-to-access-home.md)
    - [2022.11.19 Why Job Restart?](blog/2022-11-19-why-job-restart.md)


## Login

### passwordless

It is annoying to enter your password when you login the server. Here is a strategy to automatically login.

=== "Linux/Mac"

    In the command line, generate SSH key pair by running 

    ```bash
    $ ssh-keygen -t rsa
    ```

    then copy the generated public key `~/.ssh/id_rsa.pub` to the file `~/.ssh/authorized_keys` on the server (if no such file and folder, just create them). See [:link:](../servers#initial-start) for more details.

=== "Windows"

    If you use the Windows Subsystem for Linux (WSL), just follow the steps for "Linux/Mac".

    Take the PuTTY client as an example, you can use PuTTYgen to generate the ssh key pair. Here is a [tutorial](https://www.ssh.com/academy/ssh/putty/windows/puttygen).

Now, we can directly access the server via

```bash
$ ssh USERNAME@chpc-login01.itsc.cuhk.edu.hk
```

!!! Tip
    For clients such as PuTTY on Windows, just type the username and host in the specific blanks. Moreover, you can always re-express a SSH command by typing each filed in the respective blank.

Since the login node is not suggested/allowed to run your test jobs, it would be more convenient to login in the test node, `sandbox`. This can be done with consecutive ssh,

```bash
$ ssh -t USERNAME@chpc-login01.itsc.cuhk.edu.hk ssh sandbox
```

where `-t` aims to avoid the following warning

!!! warning
    Pseudo-terminal will not be allocated because stdin is not a terminal.

### bypass the login node

!!! tip "TL; DR"
    If you cannot access the cluster due to the out-of-service login node or outside campus without VPN, you can still login the cluster with
    ```bash
    $ ssh -p PORT YOUR_USERNAME@MY_BRIDGE
    ```
    If you are interested, please contact me for `PORT` and `MY_BRIDGE`. [**Buy me a coffee**](https://user-images.githubusercontent.com/13688320/173276412-22efccd3-28d4-4251-9b23-6704179d4742.jpg) if it is useful.

Usually, only the login node is out of service, but the jobs on computing nodes would not be affected. So there is a way to bypass the unaccessible login node. It also works when you are outside campus without VPN. Briefly, we can construct a tunnel (bridge) from you laptop to the cluster server via a middle machine.

If you want to use my established bridge (tunnel), the above "TL; DR" is enough, but you need to fulfill the following requirements.

!!! warning "Requirements for Using My Bridge (Tunnel)"
    You should have configured SSH key pair for passwordless login. And if you are using PuTTY, make sure your SSH key is in the standard OpenSSH format instead of PuTTY's style, otherwise it will throw the "invalid key" error. You can convert PuTTY's format in PuTTYgen, see [:link:](https://www.simplified.guide/putty/convert-ppk-to-ssh-key) for more details.


If you want to establish a tunnel by yourself, here are the detailed mechanism.

!!! warning "Requirements for Self-established Tunnel"
    You can access another **middle machine** which has a public or campus IP. Otherwise, you can try to use free tools like `ngrok` to generate a public ip for your local machine, see [my notes on how to access the intranet from outside](../../Internet/sci-int/#ngrok).

- Step 1: ssh to the middle machine from nodes except the login node of ITSC cluster, say `sandbox`, with the remote port forwarding option `-R PORT:localhost:22`
- Step 2: ssh back to `sandbox` by specifying the port `-p PORT`

!!! tip
    Sometimes ssh session might be disconnected if no further actions, so it would be necessary to replace `ssh` with [`autossh` (see my notes)](../../Internet/#autossh)

The sketch plot is as follows,

![image](https://user-images.githubusercontent.com/13688320/126862431-67358586-1f7b-410f-bafe-e3941bc71d40.png)

It is necessary to check the status of the tunnel. If the connection is broken, such as the sandbox has rebooted, pop a message window to remind to re-establish the tunnel in time.

The command to pop message is `notify-send`, and note that a successful command exits with 0, use the script to check the status of ssh connection,

```bash
--8<-- "docs/Linux/check_ssh.sh"
```

Create a regular job as follows,

```bash
$ crontab -e
0 * * * * export XDG_RUNTIME_DIR=/run/user/$(id -u); for host in sandbox STAPC ROCKY; do sh /home/weiya/github/techNotes/docs/Linux/check_ssh.sh $host; done
```

where `export XDG_RUNTIME_DIR=/run/user/$(id -u)` is necessary for `notify-send` to pop a window (refer to [Notify-send doesn't work from crontab](https://askubuntu.com/a/1098206))

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
alias sacctchpc='sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocTRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn)'
# list jobs sorted by priority (only involved stat)
alias sacctstat='sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocTRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn) | sed -n "1,2p;/stat/p"'
```

- functions 

```bash
#request specified nodes in interactive mode
request_cn() { srun -p stat -q stat -w chpc-cn1$1 --pty bash -i; }
request_gpu() { srun -p stat -q stat --gres=gpu:1 -w chpc-gpu01$1 --pty bash -i; }
request_gpu_chpc() { srun -p chpc --gres=gpu:1 -w chpc-gpu$1 --pty bash -i; }
t() { tmux a -t $1 || tmux new -s $1; }
```

## Interactive Mode

Strongly recommend the interactive mode when you debug your program or want to check the outputs of each step.

### `qsub -I`

The simplest way is

```bash
[sXXXX@chpc-login01 ~] $ qsub -I
```

If there are idle nodes, then you would be allocated to a node, and pay attention to the prompt, which indicates where you are. For example, `sXXXX@chpc-login01` means you are on the `chpc-login01` node.

Sometimes you can be automatically brought into the target node, then you are done. But sometimes it just displays the node you are allocated, such as

```bash
[sXXXX@chpc-login01 ~] $ qsub -I
...
salloc: Nodes chpc-cn011 are ready for job
[sXXXX@chpc-login01 ~] $
```

then you need to manually ssh into the target node

```bash
[sXXXX@chpc-login01 ~] $ ssh chpc-cn101
[sXXXX@chpc-cn101 ~] $
```

### `srun -w`

Sometimes you might want to use a specified node, say you want to use GPU (DO NOT forget `--gres=gpu:1`), then you can specify your node via the option `-w`. Moreover, you'd better specify the partition and QoS policy `-p stat -q stat`, which counts your quota of usage. The interactive command is specified via `--pty bash -i`.

The complete command is

```bash
[sxxxxx@chpc-login01 ~]$ srun -p stat -q stat --gres=gpu:1 -w chpc-gpu010 --pty bash -i
srun: job XXXXXX queued and waiting for resources
srun: error: Lookup failed: Unknown host
srun: job XXXXXX has been allocated resources
[sxxxxx@chpc-gpu010 ~]$ 
```

Upon you are allocated a node, you can do what you want just like on your own laptop.

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

## Exit Code 

As the [official documentation](https://slurm.schedmd.com/job_exit_code.html) said, a job's exit code (aka exit status, return code and completion code) is captured by Slurm and saved as part of the job record. For sbatch jobs, the exit code that is captured is the output of the batch script.

- Any non-zero exit code will be assumed to be a job failure and will result in a Job State of FAILED with a Reason of "NonZeroExitCode".
- The exit code is an 8 bit unsigned number ranging between 0 and 255.
- **When a signal was responsible for a job or step's termination, the signal number will be displayed after the exit code, delineated by a colon(:)**

We can check the exit code of particular jobs,

```bash
sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocTRES,NNodes,NodeList,Submit,QOS,STATE,ExitCode,DerivedExitCode
```

e.g.,

![](https://user-images.githubusercontent.com/13688320/130934797-f6908044-d583-459e-b3ea-09df362f442c.png)

where 

- the first one is a toy example and kill by myself with `kill -s 9 XX`, so the right of `:` is signal `9`, and it exits with zero code
- the second one is the one shared by [@fangda](https://github.com/songfd2018). It is exactly reversed, and I suspect that it might be due to other reasons.

see also: [3.7.6 Signals](https://www.gnu.org/software/bash/manual/html_node/Signals.html) and 

```bash
# http://www.bu.edu/tech/files/text/batchcode.txt
Name     Number (SGI)   Number (IBM)
SIGHUP      1              1
SIGINT      2              2
SIGQUIT     3              3
SIGILL      4              4
SIGTRAP     5              5
SIGABRT     6              6
SIGEMT      7              7
SIGFPE      8              8
SIGKILL     9              9
SIGBUS      10             10
SIGSEGV     11             11
SIGSYS      12             12
SIGPIPE     13             13
SIGALRM     14             14
SIGTERM     15             15
SIGUSR1     16             30
SIGUSR2     17             31
SIGPOLL     22             23
SIGIO       22             23
SIGVTALRM   28             34
SIGPROF     29             32
SIGXCPU     30             24
SIGXFSZ     31             25
SIGRTMIN    49             888
SIGRTMAX    64             999
```


## Job Priority

The submitted jobs are sorted by the calculated job priority in descending order. 

!!! tip "TL;DR"
    You can check the priority of all submitted jobs (not only yours but also others), and then you can find where you are, and figure out when your job can start to run.
    ```bash
    $ sacct -a -X --format=Priority,User%20,JobID,Account,AllocCPUS,AllocTRES,NNodes,NodeList,Submit,QOS | (sed -u 2q; sort -rn)
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

## QOS

In practice, we are always told that the maximum number of CPUs/cores and Jobs, which can be specified by QOS. Recall that we are suggested to specify `-q stat` when we submit jobs. Without such option, it uses the default one `-q normal`. Actually, they corresponds to two different Quality of Service (QOS), i.e., two different quotas.

For example, in the below configuration, when we use `-p stat -q stat`, the maximum number of jobs and cpus are limited to be 30. But if we just specify `-p stat`, the submitted jobs would be counted into `-p normal` whose limitation is only 10. (see [:link:](https://github.com/szcf-weiya/techNotes/issues/7) for my exploration)

```bash
$ sacctmgr show qos format=name,MaxJobsPU,MaxSubmitPU,MaxTRESPU
      Name MaxJobsPU MaxSubmitPU     MaxTRESPU 
---------- --------- ----------- ------------- 
    normal                    10               
      stat                    30        cpu=30 
    20jobs                    20               
        p1                    10               
        p2                    10               
        p3                    10               
      hold         0          10               
    tfchan                    10               
      bull                    50               
      ligo                   100               
      demo                    10               
yingyingw+                    30        cpu=30 
     bzhou                    10               
       geo                    10               
     cstat                              cpu=16 
```

Furthermore, the QOS configuration is defined in 

```bash
$ cat /etc/slurm/slurm.conf 
...
# Partition
PartitionName=chpc Nodes=chpc-cn[002-029,033-040,042-050],chpc-gpu[001-003],chpc-k80gpu[001-002],chpc-large-mem01,chpc-m192a[001-010] Default=YES MaxTime=7-0 MaxNodes=16 State=UP DenyAccounts=public AllowQos=normal,cstat,tfchan,ligo #DenyQos=stat,bull,demo
PartitionName=public Nodes=chpc-cn[005,015,025,035,045],chpc-gpu001 MaxTime=7-0 State=UP 
PartitionName=stat Nodes=chpc-cn[101-110],chpc-gpu[010-014] State=UP AllowAccounts=stat QOS=stat
PartitionName=yingyingwei Nodes=chpc-cn111,chpc-gpu015 State=UP AllowGroups=yingyingwei QOS=yingyingwei
PartitionName=bzhou Nodes=chpc-gpu[004-009] State=UP AllowAccounts=boleizhou QOS=bzhou
PartitionName=tjonnie Nodes=chpc-cn[030-032,041] State=UP AllowGroups=s1155137381 QOS=ligo
#PartitionName=ligo Nodes=chpc-cn050 State=UP AllowAccounts=tjonnieli QOS=ligo
#PartitionName=demo Nodes=chpc-cn049 State=UP AllowAccounts=pione QOS=demo
#PartitionName=geo Nodes=chpc-cn048 State=UP AllowGroups=s1155102420 QOS=geo
PartitionName=itsc Nodes=ALL State=UP AllowAccounts=pione QOS=bull Hidden=yes 
```

so to gain more quota, a possible (might **not friendly** if without notification) way is to try other policy `-q` on other partition `-p`. For example, the above `cstat` does not specify the limitation on the number of submit jobs (`MaxJobsPU`) and no `cstat` record (and hence no contraints like `AllowAccounts` and `AllowQos`) in the configuration file, so we can submit more than 30 jobs with `-q cstat`, althought might not be too much since it limits the resource `cpu=16` in `MaxTRESPU`. 

## CPU/Memory Usage

Check the CPU and memory usage of a specific job. The natural way is to use `top` on the node that run the job. After ssh into the corresponding node, get the map between job id and process id via

```bash
$ scontrol listpids YOUR_JOB_ID
```

Note that this only works with processes on the node on which `scontrol` is run, i.e., we cannot get the corresponding pid before ssh into the node.

Then check the results of `top` and monitor the CPU/memory usage by the job given the pid. Or explicitly specify the pid via `top -p PID_OF_JOB`

Alternatively, a more direct way is to use `sstat` command, which reports various status information, including CPU and memory, for running jobs.

```bash
$ sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,MaxRSS,MaxVMSize -j JOBID
    AveCPU   AvePages     AveRSS  AveVMSize     MaxRSS  MaxVMSize 
---------- ---------- ---------- ---------- ---------- ---------- 
 00:02.000         30      1828K    119820K      1828K    276808K 
```

Correspondingly, the result from `top` is

```bash
PID USER PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME COMMAND                                                                                                              
213435 XXX 20   0  119820   2388   1772 S   0.0  0.0   0:00.27 bash 
```

where `VIRT` == `AveVMSIZE`. The detailed meaning can be found via `man top`,

- `VIRT`: Virtual Memory Size
- `RES`: Resident Memory Size
- `%MEM`: `RES` divided by total physical memory

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

## Inherited Environment

By default, `sbatch` will inherit the environment variables, so

```bash
$ module load R/3.6
$ sbatch -p stat -q stat << EOF
> #!/bin/sh
> echo $PATH
> which R
> EOF
Submitted batch job 319113
$ cat slurm-319113.out 
/opt/share/R/3.6.3/bin:...
/opt/share/R/3.6.3/bin/R
```

we can disable the inheriting behavior via 

```bash
$ sbatch -p stat -q stat --export=NONE << EOF
> #!/bin/sh
> echo $PATH
> which R
> EOF
Submitted batch job 319110
$ cat slurm-319110.out 
/opt/share/R/3.6.3/bin:
which: no R in (...
```

But note that `$PATH` still has the path to `R/3.6`, the explanation would be that the substitution has been executed before submitting.

The detailed explanation of `--export` can be found in `man sbatch`

```bash
       --export=<[ALL,]environment variables|ALL|NONE>
              Identify  which environment variables from the submission environment are propagated to the launched applica‐
              tion. Note that SLURM_* variables are always propagated.

              --export=ALL
                        Default mode if --export is not specified. All of the users environment will be loaded (either from
                        callers environment or clean environment if --get-user-env is specified).

              --export=NONE
                        Only  SLURM_*  variables  from the user environment will be defined. User must use absolute path to
                        the binary to be executed that will define the environment.  User can not specify explicit environ‐
                        ment variables with NONE.  --get-user-env will be ignored.
                        This  option  is particularly important for jobs that are submitted on one cluster and execute on a
                        different cluster (e.g. with different paths).  To avoid steps inheriting environment  export  set‐
                        tings  (e.g.  NONE) from sbatch command, the environment variable SLURM_EXPORT_ENV should be set to
                        ALL in the job script.
```
