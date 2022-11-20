---
comments: true
---

起因是发现突然多了 slurm 的输出文件，

```bash
$ ll -at | head -10
       0 Nov 19 17:56 slurm-1144481.out
```

第一感觉是 dummy.job 超过时间限制了，然后一看

```bash
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1144481   statdgx dummy.jo s1155113  R    6:57:16      1 chpc-gpu019
```

竟然才运行 6 小时，而当前时间为

```bash
$ date
Sun Nov 20 00:53:59 HKT 2022
```

这刚好能跟 slurm-1144481.out 的更新时间对得上 (17:56 + 6:57 = 24:53)，但是 6h 前我可还在睡梦中，不可能自己提交 job。

第一个猜测是

!!! question "触发时间限制了，然后 job 自动重启了？"

      

毕竟此前没遇到过触发时间限制，所以并不确信触发时间限制后的行为。如果是这样，下面要验证的就是此 job 实际提交于 2022-10-20 左右。

首先检查 slurm-1144481.out 详细的时间戳，

```bash
$ stat slurm-1144481.out 
  File: ‘slurm-1144481.out’
  Size: 0         	Blocks: 0          IO Block: 1048576 regular empty file
Device: 2ah/42d	Inode: 3225764414  Links: 1

Context: system_u:object_r:user_home_t:s0
Access: 2022-11-02 06:23:31.643498231 +0800
Modify: 2022-11-19 17:56:05.643506847 +0800
Change: 2022-11-19 17:56:05.643506847 +0800
 Birth: -
```

早于 2022-11-02 就有过访问行为，至少表明该文件不是刚刚创建的。

进一步查看 job 提交日志，

```bash
$ sacct --format=JobID,JobName,NodeList,Submit,Start,SystemComment,End,State,Reason,Suspended,Timelimit,Comment --starttime=2022-10-01 | grep dummy
1114583       dummy.job      chpc-cn116 2022-09-29T05:29:58 2022-09-29T05:29:58                 2022-10-14T10:24:05 CANCELLED+                   None   00:00:00 30-00:00:+                
1114584       dummy.job      chpc-cn116 2022-09-29T05:37:18 2022-09-29T05:37:18                 2022-10-14T10:24:03 CANCELLED+                   None   00:00:00 30-00:00:+                
1144480       dummy.job   None assigned 2022-11-02T06:21:39 2022-11-02T06:21:39                 2022-11-02T06:21:39     FAILED                   None   00:00:00 30-00:00:+                
1144481       dummy.job     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 30-00:00:+         
```

发现相邻的上一个 job 1144480 是 2022-11-02T06:21:39 提交，这说明 1144481 应不早于这个时间戳提交。那第一个猜测不成立。

那问题就来了，

!!! question "既然没触发 30 天的时间限制，为什么会被重启？"



刚好在另一处记录了此次提交 job 提交的记录

![image](https://user-images.githubusercontent.com/13688320/202863225-f06c2678-8a5f-4dda-b2c3-daf0d0f97ad0.png)

修改时间换成 HKT 的话是 2022-11-02T06:25，刚好在 job 1144480 的 submit 的时间戳几分钟之后，

还有一点，注意到 `REASON` 一列 1144481 值为 "BeginTime"，根据[:link:](https://stackoverflow.com/questions/71921445/slurm-pending-jobs)，这应该是 pending job 的 reason， 但 job 的详细信息中 `Reason = None`，可能因为已经处于 RUNNING 状态，但是 `sacct` 还没有

```
$ scontrol show job 1144481
JobId=1144481 JobName=dummy.job

   Priority=29422 Nice=0 Account=stat QOS=stat
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=27 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=07:52:04 TimeLimit=30-00:00:00 TimeMin=N/A
   SubmitTime=2022-11-19T17:52:26 EligibleTime=2022-11-19T17:54:27
   AccrueTime=2022-11-19T17:54:27
   StartTime=2022-11-19T17:56:05 EndTime=2022-12-19T17:56:05 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2022-11-19T17:56:05
   Partition=statdgx AllocNode:Sid=chpc-sandbox:25884
   ReqNodeList=chpc-gpu019 ExcNodeList=(null)
   NodeList=chpc-gpu019
   BatchHost=chpc-gpu019
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=1,mem=16000M,node=1,billing=1,gres/gpu=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=0 MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Power=
   MemPerTres=gpu:16000
   TresPerNode=gpu:1
   NtasksPerTRES:0
```

所以还有另一种可能性，1144481 虽然被提交了，但是一直在队列中，处于 PENDING 状态，直至 2022-11-02T06:21:39 才运行。但 slurm 的提交日志确实也区分了 submit 和 start 这两个时间戳，对于 1144481，这两个时间戳都是 2022-11-19，且中间确有 4 分钟的间隔。另一方面，截图中也明确写成 1144481 已经 submitted，所以它的 submit 时间绝不应该是 2022-11-19T17:52:26. 除非...

!!! question "解除 PENDING 后，系统重新提交，产生了新的 submit 时间戳？"
     
    
于是检查是否存在 STATE 为 RUNNING 但是 REASON 也为 BeginTime 的 job，还真有几个，

```bash
$ sacct -a -X --format=JobID,JobName,NodeList,Submit,Start,SystemComment,End,State,Reason,Suspended,Timelimit,Comment --starttime=2022-10-01 | grep RUNNING | grep BeginTime
1150975           s1h71 chpc-cn[023-02+ 2022-11-17T11:43:44 2022-11-18T21:06:01                             Unknown    RUNNING              BeginTime   00:00:00 7-00:00:00                
1151170      PRD.71.02+      chpc-cn002 2022-11-17T16:36:11 2022-11-18T01:01:59                             Unknown    RUNNING              BeginTime   00:00:00 7-00:00:00                
1144481       dummy.job     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 30-00:00:+                
1150938      tc_aipw_g+     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 4-04:00:00                
1150942      1tc_aipw_+     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 4-04:00:00
```

而且注意最后三条的 submit, start 时间戳竟然一模一样，而且也在 chpc-gpu019，这不是巧合！于是根据时间戳搜索，看看还有没有更多的 job，又发现两条，但它们运行几个小时后被 cancel 了。

```bash
$ sacct -a -X --format=JobID,JobName,NodeList,Submit,Start,SystemComment,End,State,Reason,Suspended,Timelimit,Comment --starttime=2022-10-01 | grep 2022-11-19T17:52:26
1152221       p0.3_glm1 chpc-cn[053-05+ 2022-11-18T20:24:06 2022-11-18T20:24:06                 2022-11-19T17:52:26  NODE_FAIL                   None   00:00:00 7-00:00:00                
1144481       dummy.job     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 30-00:00:+                
1150938      tc_aipw_g+     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 4-04:00:00                
1150942      1tc_aipw_+     chpc-gpu019 2022-11-19T17:52:26 2022-11-19T17:56:05                             Unknown    RUNNING              BeginTime   00:00:00 4-04:00:00                
1152285      mulch_201+ chpc-large-mem+ 2022-11-19T17:52:26 2022-11-19T17:55:05                 2022-11-19T20:49:15 CANCELLED+              BeginTime   00:00:00 7-00:00:00                
1152287      nomulch_2+      chpc-cn051 2022-11-19T17:52:26 2022-11-19T17:56:05                 2022-11-19T20:49:17 CANCELLED+              BeginTime   00:00:00 7-00:00:00 
```