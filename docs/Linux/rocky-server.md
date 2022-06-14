# Rocky Linux 

Rocky Linux refers to the new server in Fan's Group

```bash
$ lsb_release -a
LSB Version:	:core-4.1-amd64:core-4.1-noarch
Distributor ID:	Rocky
Description:	Rocky Linux release 8.4 (Green Obsidian)
Release:	8.4
Codename:	GreenObsidian
```

where the distribution name `Rocky Linux` is new to me, although I might heard it before.

> [Rocky Linux](https://github.com/rocky-linux/rocky) is a community enterprise Operating System designed to be 100% bug-for-bug compatible with Enterprise Linux, now that CentOS has shifted direction.

Here is a Chinese discussion, [如何看待CentOS 创始人新发起的Rocky Linux项目?](https://www.zhihu.com/question/434205770)

## Automatic Backup: rsnapshot

Follow the tutorial [Automatic Backups on Ubuntu](https://www.cs.cornell.edu/~edward/backups.html)

1. install `yum install rsnapshot`
2. configure `/etc/rsnapshot.conf`, see the reference for more details
3. schedule backup via `cron` and `anacron`

```console
# cat /etc/anacrontab 
#period in days   delay in minutes   job-identifier   command
...
1	10	daily-snapshot	/usr/bin/rsnapshot daily
7 	20	weekly-snapshot	/usr/bin/rsnapshot weekly

# cat /etc/cron.d/rsnapshot 
0 */4 * * * root /usr/bin/rsnapshot hourly
```

!!! info
    `cron` and `anacron` are two different utilities that can automatically run a program at scheduled times.
    - `cron` is older, and it will not run any scheduled task whose scheduled time occurs while the computer is powered off.
    - `anacron` will re-run the missed task the next time the computer turns on.    

## Install Slurm

Different from CHPC's cluster, the group server is just one node. So first of all, I search the possibility and necessity of slurm on single node, here are some references.

- [SLURM single node install](http://docs.nanomatch.de/technical/SimStackRequirements/SingleNodeSlurm.html#slurm-single-node-install)
- [Setting up a single server SLURM cluster](https://rolk.github.io/2015/04/20/slurm-cluster)

Comparing with the standard installation, it seems no much difference, so I follow the official [Quick Start Administrator Guide](https://slurm.schedmd.com/quickstart_admin.html) to install.

- install munge: the first attempt is not enough, and its `devel` will be also installed

```bash
yum install munge
```

- download source package

```bash
wget https://download.schedmd.com/slurm/slurm-20.11.7.tar.bz2
```

- build it

```console
# rpmbuild -ta slurm*.tar.bz2
```

but it throws that

```bash
error: Failed build dependencies:
	munge-devel is needed by slurm-20.11.7-1.el8.x86_64
	pam-devel is needed by slurm-20.11.7-1.el8.x86_64
	perl(ExtUtils::MakeMaker) is needed by slurm-20.11.7-1.el8.x86_64
	readline-devel is needed by slurm-20.11.7-1.el8.x86_64
```

so I need to resolve these dependencies firstly.

- install `munge-devel`

firstly I tried 

```console
# yum install munge-devel
Last metadata expiration check: 1:10:07 ago on Sat 26 Jun 2021 06:42:42 PM HKT.
No match for argument: munge-devel
Error: Unable to find a match: munge-devel
```

!!! tip "use console for root's `#`"
    In bash code block, the symbol `#` would be treated as comments, but I want to indicate the operations are performed by root, so change `bash` to `console` refer to [Markdown syntax for interactive shell commands](https://groups.google.com/g/nikola-discuss/c/5AzGQjlhB8g)

Reminded by [Slurm does not install from the epel repo](https://forums.rockylinux.org/t/slurm-does-not-install-from-the-epel-repo/2832), I tried to enable the `powertools` repo following the instruction, [How to enable PowerTools repository in CentOS 8?](https://serverfault.com/questions/997896/how-to-enable-powertools-repository-in-centos-8)

```console
# yum repolist 
repo id                                                                repo name
appstream                                                              Rocky Linux 8 - AppStream
baseos                                                                 Rocky Linux 8 - BaseOS
epel                                                                   Extra Packages for Enterprise Linux 8 - x86_64
epel-modular                                                           Extra Packages for Enterprise Linux Modular 8 - x86_64
extras                                                                 Rocky Linux 8 - Extras

# yum install dnf-plugins-core
Last metadata expiration check: 1:15:02 ago on Sat 26 Jun 2021 06:42:42 PM HKT.
Package dnf-plugins-core-4.0.18-4.el8.noarch is already installed.
Dependencies resolved.
Nothing to do.
Complete!

# yum config-manager --set-enabled powertools
# yum repolist 
repo id                                                                repo name
appstream                                                              Rocky Linux 8 - AppStream
baseos                                                                 Rocky Linux 8 - BaseOS
epel                                                                   Extra Packages for Enterprise Linux 8 - x86_64
epel-modular                                                           Extra Packages for Enterprise Linux Modular 8 - x86_64
extras                                                                 Rocky Linux 8 - Extras
powertools                                                             Rocky Linux 8 - PowerTools
```

then I can install `munge-devel`,

```console
# yum install munge-devel
Rocky Linux 8 - PowerTools                                                                                                                              2.6 MB/s | 2.1 MB     00:00    
Dependencies resolved.
========================================================================================================================================================================================
 Package                                      Architecture                            Version                                         Repository                                   Size
========================================================================================================================================================================================
Installing:
 munge-devel                                  x86_64                                  0.5.13-2.el8                                    powertools                                   27 k

Transaction Summary
========================================================================================================================================================================================
Install  1 Package

Total download size: 27 k
Installed size: 23 k
Is this ok [y/N]: y
Downloading Packages:
munge-devel-0.5.13-2.el8.x86_64.rpm                                                                                                                     716 kB/s |  27 kB     00:00    
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total                                                                                                                                                    95 kB/s |  27 kB     00:00     
Running transaction check
Transaction check succeeded.
Running transaction test
Transaction test succeeded.
Running transaction
  Preparing        :                                                                                                                                                                1/1 
  Installing       : munge-devel-0.5.13-2.el8.x86_64                                                                                                                                1/1 
  Running scriptlet: munge-devel-0.5.13-2.el8.x86_64                                                                                                                                1/1 
  Verifying        : munge-devel-0.5.13-2.el8.x86_64                                                                                                                                1/1 
Installed products updated.

Installed:
  munge-devel-0.5.13-2.el8.x86_64                                                                                                                                                       

Complete!
```

- install other dependencies,

```console
# yum install pam-devel
# yum install readline-devel
# yum install perl-ExtUtils-MakeMaker.noarch
```

where the package name for the last one is determining by simply googling `perl(ExtUtils::MakeMaker)`

- build again

```console
# rpmbuild -ta slurm-20.11.7.tar.bz2
Processing files: slurm-slurmdbd-20.11.7-1.el8.x86_64
error: File not found: /root/rpmbuild/BUILDROOT/slurm-20.11.7-1.el8.x86_64/usr/lib64/slurm/accounting_storage_mysql.so


RPM build errors:
    Macro expanded in comment on line 22: %_prefix path		install path for commands, libraries, etc.

    Macro expanded in comment on line 170: %define _unpackaged_files_terminate_build      0

    Empty %files file /root/rpmbuild/BUILD/slurm-20.11.7/slurm.files
    File not found: /root/rpmbuild/BUILDROOT/slurm-20.11.7-1.el8.x86_64/usr/lib64/slurm/accounting_storage_mysql.so
    File listed twice: /usr/lib/.build-id/13/1f6904421dbe39c72395ad1902cee51db6a0ec
    File listed twice: /usr/lib/.build-id/28/be902644d9888579406edd90f00c5f9c9c2aa6
    File listed twice: /usr/lib/.build-id/7c/791fea05f7db7b3de616553cd27fa82b3fc510
    File listed twice: /usr/lib/.build-id/93/bbb15fe8897a7d4745fc56dcb3e5435f80bf4c
    Deprecated external dependency generator is used!
    Deprecated external dependency generator is used!
    Deprecated external dependency generator is used!
    Empty %files file /root/rpmbuild/BUILD/slurm-20.11.7/example.configs
    Deprecated external dependency generator is used!
    Deprecated external dependency generator is used!
    Deprecated external dependency generator is used!
    File not found: /root/rpmbuild/BUILDROOT/slurm-20.11.7-1.el8.x86_64/usr/lib64/slurm/accounting_storage_mysql.so
```

which means that `mysql` is needed. 

- install `mariadb` and its `devel` following [How to Install Slurm on CentOS 7 Cluster](https://www.slothparadise.com/how-to-install-slurm-on-centos-7-cluster/)

```bash
# yum install mariadb-server mariadb-devel
```

- build again

```bash
# rpmbuild -ta slurm-20.11.7.tar.bz2
Wrote: /root/rpmbuild/RPMS/x86_64/slurm-openlava-20.11.7-1.el8.x86_64.rpm
Wrote: /root/rpmbuild/RPMS/x86_64/slurm-contribs-20.11.7-1.el8.x86_64.rpm
Wrote: /root/rpmbuild/RPMS/x86_64/slurm-pam_slurm-20.11.7-1.el8.x86_64.rpm
Executing(%clean): /bin/sh -e /var/tmp/rpm-tmp.52rp8H
+ umask 022
+ cd /root/rpmbuild/BUILD
+ cd slurm-20.11.7
+ rm -rf /root/rpmbuild/BUILDROOT/slurm-20.11.7-1.el8.x86_64
```

it succeeds!

- install rpms

all built rpms are

```bash
# ls rpmbuild/RPMS/x86_64/
slurm-20.11.7-1.el8.x86_64.rpm                  slurm-libpmi-20.11.7-1.el8.x86_64.rpm     slurm-slurmctld-20.11.7-1.el8.x86_64.rpm
slurm-contribs-20.11.7-1.el8.x86_64.rpm         slurm-openlava-20.11.7-1.el8.x86_64.rpm   slurm-slurmd-20.11.7-1.el8.x86_64.rpm
slurm-devel-20.11.7-1.el8.x86_64.rpm            slurm-pam_slurm-20.11.7-1.el8.x86_64.rpm  slurm-slurmdbd-20.11.7-1.el8.x86_64.rpm
slurm-example-configs-20.11.7-1.el8.x86_64.rpm  slurm-perlapi-20.11.7-1.el8.x86_64.rpm    slurm-torque-20.11.7-1.el8.x86_64.rpm

# yum --nogpgcheck localinstall slurm-20.11.7-1.el8.x86_64.rpm slurm-devel-20.11.7-1.el8.x86_64.rpm slurm-perlapi-20.11.7-1.el8.x86_64.rpm slurm-torque-20.11.7-1.el8.x86_64.rpm slurm-slurmctld-20.11.7-1.el8.x86_64.rpm slurm-slurmd-20.11.7-1.el8.x86_64.rpm
```

- then configure from [https://slurm.schedmd.com/configurator.easy.html](https://slurm.schedmd.com/configurator.easy.html)

change the hostname to `hostname -s`

- be careful about the path to the logfile,

```
mkdir /var/spool/slurmctld
chown slurm: /var/spool/slurmctld
chmod 755 /var/spool/slurmctld
touch /var/log/slurmctld.log
chown slurm: /var/log/slurmctld.log
touch /var/log/slurm_jobacct.log /var/log/slurm_jobcomp.log
chown slurm: /var/log/slurm_jobacct.log /var/log/slurm_jobcomp.log
```

also

```bash
mkdir /var/spool/slurmd
chown slurm: /var/spool/slurmd
chmod 755 /var/spool/slurmd
touch /var/log/slurmd.log
chown slurm: /var/log/slurmd.log
```

then check 

```bash
# slurmd -C
NodeName=stapcXXX CPUs=96 Boards=1 SocketsPerBoard=2 CoresPerSocket=24 ThreadsPerCore=2 RealMemory=385333
UpTime=1-04:02:46
```

- check if `sinfo` OK,

```console
# sinfo 
slurm_load_partitions: Unable to contact slurm controller (connect failure)
# scontrol show nodes
slurm_load_node error: Unable to contact slurm controller (connect failure)
```

and the service `slurmd` and `slurmctld` fails to start,

```console
# systemctl status slurmd.service
● slurmd.service - Slurm node daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmd.service; enabled; vendor preset: disabled)
   Active: failed (Result: exit-code) since Sat 2021-06-26 20:51:09 HKT; 4s ago
  Process: 126781 ExecStart=/usr/sbin/slurmd -D $SLURMD_OPTIONS (code=exited, status=1/FAILURE)
 Main PID: 126781 (code=exited, status=1/FAILURE)

Jun 26 20:51:09 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Slurm node daemon.
Jun 26 20:51:09 stapcXXX.sta.cuhk.edu.hk systemd[1]: slurmd.service: Main process exited, code=exited, status=1/FAILURE
Jun 26 20:51:09 stapcXXX.sta.cuhk.edu.hk systemd[1]: slurmd.service: Failed with result 'exit-code'.

# systemctl status slurmctld.service
● slurmctld.service - Slurm controller daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmctld.service; enabled; vendor preset: disabled)
   Active: failed (Result: exit-code) since Sat 2021-06-26 20:52:26 HKT; 4s ago
  Process: 126814 ExecStart=/usr/sbin/slurmctld -D $SLURMCTLD_OPTIONS (code=exited, status=1/FAILURE)
 Main PID: 126814 (code=exited, status=1/FAILURE)

Jun 26 20:52:26 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Slurm controller daemon.
Jun 26 20:52:26 stapcXXX.sta.cuhk.edu.hk systemd[1]: slurmctld.service: Main process exited, code=exited, status=1/FAILURE
Jun 26 20:52:26 stapcXXX.sta.cuhk.edu.hk systemd[1]: slurmctld.service: Failed with result 'exit-code'.
```

the error turns to be the incosistent path in the config file, `slurm.conf`

```bash
StateSaveLocation: /var/spool/slurmctld
```

and then

```bash
mkdir /var/spool/slurmctld
chown slurm: /var/spool/slurmctld
```

but the default path is `/var/spool`. 

- run `slurmctld` service

After correcting the path, the service begins to work,

```console
# systemctl status slurmctld.service 
● slurmctld.service - Slurm controller daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmctld.service; enabled; vendor preset: disabled)
   Active: active (running) since Sat 2021-06-26 22:11:14 HKT; 2min 18s ago
 Main PID: 128214 (slurmctld)
    Tasks: 7
   Memory: 4.4M
   CGroup: /system.slice/slurmctld.service
           └─128214 /usr/sbin/slurmctld -D

Jun 26 22:11:14 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Slurm controller daemon.
```

- run `slurmd` service

However, `slurmd` still failed, but no informative message returned by the `systemctl` command. Try to directly run the command

```console
# slurmd -D
slurmd: error: Node configuration differs from hardware: CPUs=1:96(hw) Boards=1:1(hw) SocketsPerBoard=1:2(hw) CoresPerSocket=1:24(hw) ThreadsPerCore=1:2(hw)
slurmd: error: cgroup namespace 'freezer' not mounted. aborting
slurmd: error: unable to create freezer cgroup namespace
slurmd: error: Couldn't load specified plugin name for proctrack/cgroup: Plugin init() callback failed
slurmd: error: cannot create proctrack context for proctrack/cgroup
slurmd: error: slurmd initialization failed
```

then correct the corresponding configuration. And temporarly disable cgroup by setting

```bash
Process Tracking: LinuxProc
```

as suggested in [SLURM single node install](http://docs.nanomatch.de/technical/SimStackRequirements/SingleNodeSlurm.html)

Also pay attention to `restart`, or `stop + start` the service, sometimes the configuration file might still be outdated. For example, I had set the hostname as `hostname -f` and `hostname -s`, but sometimes the hostname has not been updated after changing the hostname, so make sure the updated configuration file has been taken effects.

After necessary correction and restart,

```console
# slurmd -D
slurmd: slurmd version 20.11.7 started
slurmd: slurmd started on Sat, 26 Jun 2021 22:28:01 +0800
slurmd: CPUs=96 Boards=1 Sockets=2 Cores=24 Threads=2 Memory=385333 TmpDisk=1023500 Uptime=107065 CPUSpecList=(null) FeaturesAvail=(null) FeaturesActive=(null)
^Cslurmd: Slurmd shutdown completing
```

which indiates that now slurmd works well, then

```console
# systemctl restart slurmd.service 
# systemctl status slurmd.service 
● slurmd.service - Slurm node daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmd.service; enabled; vendor preset: disabled)
   Active: active (running) since Sat 2021-06-26 22:28:23 HKT; 1s ago
 Main PID: 129051 (slurmd)
    Tasks: 2
   Memory: 1.3M
   CGroup: /system.slice/slurmd.service
           └─129051 /usr/sbin/slurmd -D

Jun 26 22:28:23 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Slurm node daemon.
```

- accounting system

Try to start the accounting system,

```console
# sacctmgr -i add cluster stapcXXX
You are not running a supported accounting_storage plugin
Only 'accounting_storage/slurmdbd' is supported.
```

the reason is that the corresponding rpm has not been installed, and also `slurm.conf` should set

```bash
AccountingStorageType=accounting_storage/slurmdbd
```

and 

```bash
# yum --nogpgcheck localinstall slurm-slurmdbd-20.11.7-1.el8.x86_64.rpm
```

then check if it is ready,

```bash
# systemctl start slurmdbd.service 
# systemctl status slurmdbd.service 
● slurmdbd.service - Slurm DBD accounting daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmdbd.service; enabled; vendor preset: disabled)
   Active: inactive (dead)
Condition: start condition failed at Sun 2021-06-27 10:38:39 HKT; 2s ago
           └─ ConditionPathExists=/etc/slurm/slurmdbd.conf was not met
```

which means that we also need `slurmdbd.conf`.

- config `slurmdbd.conf`

copy the sample from [slurmdbd.conf](https://slurm.schedmd.com/slurmdbd.conf.html)

- config mysql

following [DaisukeMiyamoto/setup_slurm_accounting_parallelcluster.sh](https://gist.github.com/DaisukeMiyamoto/d1dac9483ff0971d5d9f34000311d312), set mariadb as follows,

```console
# mysql -u root -e "create user 'slurm'@'localhost' identified by 'xxxxxxxxxx'; grant all on slurm_acct_db.* TO 'slurm'@'localhost'; create database slurm_acct_db;"
```

then tried to run

```console
# slurmdbd -D
slurmdbd: error: mysql_real_connect failed: 2002 Can't connect to local MySQL server through socket '/var/lib/mysql/mysql.sock' (2)
slurmdbd: error: The database must be up when starting the MYSQL plugin.  Trying again in 5 seconds.
^C
```

so mysql has not been started,

```console
# systemctl enable mariadb
Created symlink /etc/systemd/system/mysql.service → /usr/lib/systemd/system/mariadb.service.
Created symlink /etc/systemd/system/mysqld.service → /usr/lib/systemd/system/mariadb.service.
Created symlink /etc/systemd/system/multi-user.target.wants/mariadb.service → /usr/lib/systemd/system/mariadb.service.
# systemctl start mariadb
```

then run again

```console
# slurmdbd -D
slurmdbd: accounting_storage/as_mysql: _check_mysql_concat_is_sane: MySQL server version is: 10.3.28-MariaDB
slurmdbd: error: Database settings not recommended values: innodb_buffer_pool_size innodb_lock_wait_timeout
slurmdbd: slurmdbd version 20.11.7 started
^Cslurmdbd: Terminate signal (SIGINT or SIGTERM) received
```

refer to [Slurm database -- MySQL configuration](https://wiki.fysik.dtu.dk/niflheim/Slurm_database#id5), write the following

```bash
[mysqld]
innodb_buffer_pool_size=1024M
innodb_log_file_size=64M
innodb_lock_wait_timeout=900
```

into file `/etc/my.cnf.d/innodb.cnf`, then restart 

```console
# systemctl stop mariadb
# systemctl start mariadb
```

now it is OK to run

```console
# slurmdbd -D
slurmdbd: accounting_storage/as_mysql: _check_mysql_concat_is_sane: MySQL server version is: 10.3.28-MariaDB
slurmdbd: slurmdbd version 20.11.7 started
^Cslurmdbd: Terminate signal (SIGINT or SIGTERM) received
```

then satrt the service

```bash
# systemctl enable slurmdbd
# systemctl start slurmdbd
# systemctl status slurmdbd
● slurmdbd.service - Slurm DBD accounting daemon
   Loaded: loaded (/usr/lib/systemd/system/slurmdbd.service; enabled; vendor preset: disabled)
   Active: active (running) since Sun 2021-06-27 10:58:27 HKT; 3s ago
 Main PID: 190603 (slurmdbd)
    Tasks: 5
   Memory: 1.8M
   CGroup: /system.slice/slurmdbd.service
           └─190603 /usr/sbin/slurmdbd -D

Jun 27 10:58:27 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Slurm DBD accounting daemon.
```

- set up qos

need to add account and user, otherwise the users are outside the control,

```console
# sacctmgr add account project Description="Projects" Organization=project
# for i in {01..14}; do sacctmgr -i create user name=project$i DefaultAccount=project ; done
```

then set the maximum submit jobs per user,

```console
# sacctmgr modify qos normal set maxsubmitjobsperuser=90
```

remove the restriction via setting a larger number (how to directly remove the constraint)

```console
# sacctmgr modify qos normal set maxsubmitjobsperuser=10000
```

together with the number of cores,

```console
# sacctmgr modify qos normal set GrpTRES=cpu=90
```

```console
sacctmgr modify qos normal set MaxTRESPerUser=cpu=80
```

but then I found that the submitted jobs are out of control. Then I realized that the account and user associations only take effect after enabling

```console
AccountingStorageEnforce=limits
```

according to [Slurm accounting -- Resource Limits](https://wiki.fysik.dtu.dk/niflheim/Slurm_accounting). Then for sure, restart `slurmd`, `slurmctld`, and `slurmdbd`, then the policy starts to work.

Then found that after changing the policy, no need to restart any service.

## CGroup

Inspired by [Setting up a single server SLURM cluster -- Setting up Control Groups](https://rolk.github.io/2015/04/20/slurm-cluster), I tried to add CGroup to control the process directly run from ssh session.

- install cgroup

following [Cgroups : Install](https://www.server-world.info/en/note?os=CentOS_7&p=cgroups&f=1)

```console
# yum -y install libcgroup libcgroup-tools
# systemctl enable cgconfig
# systemctl start cgconfig
# systemctl status cgconfig
● cgconfig.service - Control Group configuration service
   Loaded: loaded (/usr/lib/systemd/system/cgconfig.service; enabled; vendor preset: disabled)
   Active: active (exited) since Sun 2021-06-27 11:10:27 HKT; 7s ago
  Process: 191469 ExecStart=/usr/sbin/cgconfigparser -l /etc/cgconfig.conf -s 1664 (code=exited, status=0/SUCCESS)
 Main PID: 191469 (code=exited, status=0/SUCCESS)

Jun 27 11:10:27 stapcXXX.sta.cuhk.edu.hk systemd[1]: Starting Control Group configuration service...
Jun 27 11:10:27 stapcXXX.sta.cuhk.edu.hk systemd[1]: Started Control Group configuration service.
```

here is an interesting finding. Different from other services, `active (running)`, it shows `active (exited)`, more details refer to the discussion [systemctl status active: exited vs running](https://askubuntu.com/questions/1304142/systemctl-status-active-exited-vs-running).

then write `cgconfig.conf` and `cgrules.conf`, and then tried to restart `cgconfig`, but it seems no idea to judge if the setting takes effects.

Then I got to know that there is another [`cpuset` parameter](https://docs.oracle.com/en/operating-systems/oracle-linux/6/adminsg/ol_cpuset_cgroups.html) which can specify a list of CPU cores to which a group has access. Then I tried to add such parameter to `cgconfig.conf`,

```bash
cpuset {
    cpuset.cpus = "1";
}
```

and also update the corresponding `cgrules.conf`. Then I open an R session to run an expensive operation,

```r
> a = matrix(rnorm(100000000),nrow=10000)
> replicate(100, a%*%a)
```

and open another session with the same user. The target is to check if these two sessions are limited to a single core. However, each of them reach around 100% CPU.

Tried to install PAM-cgroup as also mentioned in [4.3. PER-GROUP DIVISION OF CPU AND MEMORY RESOURCES -- Procedure 4.4. Using a PAM module to move processes to cgroups](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-cpu_and_memory-use_case#proc-cpu_and_mem)

but still not work.

To validate if cpu and memory work, also run the same R script in the root session. According to `cgrules.conf`, the root session should be faster since it does not have additional restriction, but no differences.

In the checking procedure, I learn to

- find the cgroup of a process, and no `interactive` shows, [How to find out cgroup of a particular process?](https://serverfault.com/questions/560206/how-to-find-out-cgroup-of-a-particular-process/899248)

```console
# systemctl status 3378 | grep CGroup
# cat /proc/3378/cgroup 
```

- list cgroup in top: press `f`, then select `cgroup`
- pure list cgroup as in top: `systemd-cgtop`
- check parameter, `cgget -r cpuset.cpus` interactive: [10.8 Displaying and Setting Subsystem Parameters](https://docs.oracle.com/en/operating-systems/oracle-linux/6/admin/ol_getset_param_cgroups.html)

Maybe I need to change another way, and I saw `set-property` in [深入理解 Linux Cgroup 系列（一）：基本概念](https://zhuanlan.zhihu.com/p/74299886)

Then I tried to set the CPUQuota for user via (refer to [systemd, per-user cpu and/or memory limits](https://serverfault.com/questions/874274/systemd-per-user-cpu-and-or-memory-limits))

```console
# mkdir -p /etc/systemd/system/user-.slice.d
```

then write

```bash
[Slice]
CPUQuota=20%
```

into `cpu.conf`, which means that the executed processes never get more than 20% CPU time on a single CPU (refer to [2.3. MODIFYING CONTROL GROUPS](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/resource_management_guide/sec-modifying_control_groups)), then 

```console
# systemctl daemon-reload
```

Repeat the above R script, the CPU percentage would be limited to 20%, not only regular user, but also root, since root is `user-0`, such as 

```console
# ls /sys/fs/cgroup/systemd/user.slice
cgroup.clone_children  cgroup.procs  notify_on_release  tasks  user-0.slice  user-1002.slice
```

but it would not affected jobs submitted via slurm, which has been validated. So this way would be quite good.


## SelectTypeParameters

The default setting is 

```bash
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
```

the configuration of computing node is 

```bash
NodeName=stapc051 CPUs=96 Sockets=2 CoresPerSocket=24 ThreadsPerCore=2 State=UNKNOWN
```

With this setting, each job would automatically count 2 virtually CPUs, although we only request one CPU.

After changing `CR_Core` to `CR_CPU`, each job would count 1 virtually CPU.

However, the number of running jobs is still only 48, while the limits are `GrpTRES=cpu=90` and `MaxTRESPerUser=cpu=90`. 

My first guess is the scheduling system does not update on time, since if I count the completed jobs in `sacct -a -X`, the total number of cores coincidently equals to 90, then I tried to set a larger limit but not work.

Ocassionally, I found how to unset the limit, use `-1`, such as [Resource Limits](https://slurm.schedmd.com/resource_limits.html)

```bash
sacctmgr modify user bob set GrpTRES=cpu=-1,mem=-1,gres/gpu=-1
```

Notice that 48 is a special number, `96 / 2 = 48`, so my second guess is that each running job is still occupying a core, even though it indeed only counts a virtual CPU.

Check the potential related configurations,

- TaskPlugin: change the default `task/none` to `task/affinity`, since there are some related materials, [Support for Multi-core/Multi-thread Architectures](https://slurm.schedmd.com/mc_support.html), mentioned it. But not work.
- `--hint=nomultithread`: this is not a system configuration, but it can also cause only 48 jobs are running, then I have a better idea on the allocation on multi-thread machine, and here is an official example, [EXAMPLE 9: CONSUMABLE RESOURCES ON MULTITHREADED NODE, ALLOCATING ONLY ONE THREAD PER CORE](https://slurm.schedmd.com/cpu_management.html), and some examples in the mail list, [Bug 4010 - Exploring Hyper-Threading CPU nodes](https://bugs.schedmd.com/show_bug.cgi?id=4010)

Recheck the explanation of `CR_CPU`, here is one more sentence,

>  The node's Boards, Sockets, CoresPerSocket and ThreadsPerCore may optionally be configured and result in job allocations which have improved locality; however doing so will prevent more than one job from being allocated on each core.

I remembered the specification `Sockets=2 CoresPerSocket=24 ThreadsPerCore=2` is required under the default `CR_Core`, otherwise it would throw errors. But according to this warning, it would prevent twos job from being allocated on one core. Then I tried to remove this specification, and then the number of running jobs is not only 48!

!!! tip
    get pid of slurm job
    ```console
    # scontrol listpids | grep 1261
    ```
    then check the allow cpu list as in [Bug 4010 - Exploring Hyper-Threading CPU nodes](https://bugs.schedmd.com/show_bug.cgi?id=4010),
    before removing the specification, it shows
    ```console
    # cat /proc/2636286/status | grep -i cpus_allowed_list
    Cpus_allowed_list:	0-95
    ```
    but after working normally, 
    ```console
    # cat /proc/2636286/status | grep -i cpus_allowed_list
    Cpus_allowed_list:	43
    ```
    so it can be an indicator to check whether the job is allocated normally.

## PrivateData

Currently every user can view all users' job information via `squeue`. 

Set 

```bash
PrivateData=jobs
```

in `slurm.conf`.

And here are some other fields can be appended, such as `usage`, `users`, refer to [slurm.conf - Slurm configuration file](https://slurm.schedmd.com/slurm.conf.html) or `man slurm.conf` for more details.

## Diagnose Network Speed

On 6/30/21 3:16 PM, found that the network speed on the new machine seems abnormal.

> I am downloading a file via `wget`, but the speed is quite slow, just ~500kb/s, while it can achieve ~4MB/s when I try to download it from our old server

Then for complete comparisons, use the script

```bash
curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -
```

to test the network speed (refer to [How to check Internet Speed via Terminal? - Ask Ubuntu](https://askubuntu.com/questions/104755/how-to-check-internet-speed-via-terminal)

Also repeat the test on the new server to reduce random noise. The results show that both download and upload speed on the new server are much slower than the old group server and my personal laptop, ~90Mbit/s vs ~350Mbit/s.

![](https://user-images.githubusercontent.com/13688320/130889726-6168fff7-96c8-4ace-848c-587592e3075b.png)

Notice that

```bash
$ dmesg -T | grep enp
[Mon Jun 28 14:30:59 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Up 1000 Mbps Full Duplex, Flow Control: RX
[Mon Jun 28 14:31:01 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Down
[Mon Jun 28 14:31:26 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Up 100 Mbps Full Duplex, Flow Control: RX
[Mon Jun 28 14:31:26 2021] igb 0000:02:00.0 enp2s0: Link Speed was downgraded by SmartSpeed
[Mon Jun 28 15:34:48 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Down
[Mon Jun 28 15:35:13 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Up 100 Mbps Full Duplex, Flow Control: RX
[Mon Jun 28 15:35:13 2021] igb 0000:02:00.0 enp2s0: Link Speed was downgraded by SmartSpeed
[Mon Jun 28 15:45:47 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Down
[Mon Jun 28 15:46:13 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Up 100 Mbps Full Duplex, Flow Control: RX
[Mon Jun 28 15:46:13 2021] igb 0000:02:00.0 enp2s0: Link Speed was downgraded by SmartSpeed
```

where it reports that NIC link is downgraded from 1000 Mbps to 100 Mbps by SmartSpeed, which might explain that previous the fastest speed is less than 100Mbit/s.

!!! tip
    `dmesg` 用于显示开机信息。kernel 会将开机信息储存在 ring buffer 中，也保存在 `/var/log/dmesg` 文件中（参考 [Linux dmesg 命令 | 菜鸟教程](https://www.runoob.com/linux/linux-comm-dmesg.html)）。
    
    而 `man dmesg` 说该命令用于 `print or control the kernel ring buffer`，所以能够互相印证，但是 ring buffer 是否只存开机信息？
    
    - `-T` 表示以可读的时间戳显示。

    默认时间戳为 “time in seconds since the kernel started”，这也是 `dmesg` 文件中的默认格式。如果支持 `-T`，自然是很方便的，否则可能需要写脚本转换，详见 [How do I convert dmesg timestamp to custom date format? - Stack Overflow](https://stackoverflow.com/questions/13890789/how-do-i-convert-dmesg-timestamp-to-custom-date-format)

After moving to the new place, the speed seems much faster (8/16/21, 10:46 AM). The speed can achieve 500 Mbit/s (Upload) and 800 Mbit/s (Download), while previously just 90 Mbit/s. And

```bash
$ dmesg -T | grep enp
[Mon Aug 16 10:14:58 2021] igb 0000:02:00.0 enp2s0: renamed from eth0
[Mon Aug 16 10:14:58 2021] e1000e 0000:00:1f.6 enp0s31f6: renamed from eth0
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp0s31f6: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp0s31f6: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp2s0: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp2s0: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp0s31f6: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp2s0: link is not ready
[Mon Aug 16 10:15:05 2021] IPv6: ADDRCONF(NETDEV_UP): enp2s0: link is not ready
[Mon Aug 16 10:15:10 2021] e1000e 0000:00:1f.6 enp0s31f6: NIC Link is Up 1000 Mbps Full Duplex, Flow Control: None
[Mon Aug 16 10:15:10 2021] IPv6: ADDRCONF(NETDEV_CHANGE): enp0s31f6: link becomes ready
[Mon Aug 16 10:16:26 2021] e1000e 0000:00:1f.6 enp0s31f6: NIC Link is Down
[Mon Aug 16 10:16:30 2021] igb 0000:02:00.0 enp2s0: igb: enp2s0 NIC Link is Up 1000 Mbps Full Duplex, Flow Control: RX
[Mon Aug 16 10:16:31 2021] IPv6: ADDRCONF(NETDEV_CHANGE): enp2s0: link becomes ready
```

it stays at 1000 Mbps, which seems to support the testing results.

Alternatively, we can use 

```bash
$ ethtool enp2s0 | grep Speed
	Speed: 1000Mb/s
```

or

```bash
$ cat /sys/class/net/enp2s0/speed
1000
```

to obtain the speed of the NIC. It is -1 for unused interfaces. Refer to [How do I verify the speed of my NIC? - Server Fault](https://serverfault.com/questions/207474/how-do-i-verify-the-speed-of-my-nic)