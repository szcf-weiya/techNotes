---
comments: true
---

# fail to access `~`


!!! info
    The post can date back to [May 12, 2021](https://github.com/szcf-weiya/techNotes/commit/d056c5b6f46cd1b21e73c07f11aaac26edc3798a)


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

其中 `${A-B}` 的语法是如果 `A` 没有设置，则令 `B` 为 `A`，注意[其与 `${A:-B}` 的区别](../../shell/#default-value)。

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
