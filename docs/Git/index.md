# Git Tips

## 教程

1. [阮一峰的网络日志](http://www.ruanyifeng.com/blog/2014/06/git_remote.html)
2. [这些GIT经验够你用一年了](https://zhuanlan.zhihu.com/p/22666153)

## 删除远程分支

```git
git branch -r -d origin/branch-name
```
此时需要更改仓库的默认分支，不然直接运行下面的命令会报错
```
git push origin :branch-name
```

![](error_branch.png)

图中第一次是更改默认分支前的出错信息，第二次是更改完默认分支的信息。

## 删除本地分支

```git
git branch -d branch-name
```

## 提高git clone速度

```git
git config --global http.postBuffer 524288000
```

## git clone所有远程分支

```git
git clone ....
cd ..
git branch -a
git checkout -b gh-pages origin/gh-pages
```

## git删除大文件

[cnblog](http://www.cnblogs.com/lout/p/6111739.html)

## 初次配置Git
[reference](http://www.open-open.com/lib/view/open1428900970330.html)

1. 下载安装
```
apt-get install git
```

2. 配置
```
git config --global user.name "test"
git config --global user.email "test@163.com"
```

3. ssh
```
ssh-keygen -t rsa -C "test@163.com"
```

复制~/.ssh/id_rsa.pub到github上。

## 修改origin
```
git remote rm origin
git remote add origin git@192.168.1.18:mStar/OTT-dual/K3S/supernova
```

or
```
git remote set-url origin git@192.168.1.18:mStar/OTT-dual/K3S/supernova
```

## gitignore 失效

[.gitignore](http://www.pfeng.org/archives/840)

有时候在项目开发过程中，突然心血来潮想把某些目录或文件加入忽略规则，按照上述方法定义后发现并未生效，原因是.gitignore只能忽略那些原来没有被track的文件，如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。那么解决方法就是先把本地缓存删除（改变成未track状态），然后再提交：

```
git rm -r --cached .
git add .
git commit -m 'update .gitignore'
```

## 更新远程代码到本地

### 方式一
```
git remote -v
git fetch origin master
git log -p master origin master
git merge origin master
```
### 方式二
```
git fetch origin master:temp
git diff temp
git merge temp
git branch temp
```

## 关于LICENSE的选择

[阮一峰的网络日志](http://www.ruanyifeng.com/blog/2011/05/how_to_choose_free_software_licenses.html)


## git clone 某个分支或所有分支

[git clone](http://blog.csdn.net/a513322/article/details/46998325)

```
git clone -b BRANCH_NAME ...
```

or
```
git clone ...
git branch -r
git checkout BRANCH_NAME
```

## 更改远程仓库的名字

举个例子，如将一个名为epi的仓库改名为depi，再次在本地提交虽然也能成功，但是会提示你原始的仓库已经移动，请修改为新的仓库地址，于是我们可以利用下面的命令进行修改

```
git remote set-url origin git@github.com:szcf-weiya/depi.git
```

## git pull

在使用rstudio的git功能时，某次commit的时候，勾选了amend previous commit，然后push的时候就出错了

![](error_pull.PNG)

后来直接运行`git pull`后，重新push，便解决了问题。

附上git pull的某篇博客[git pull命令](http://www.yiibai.com/git/git_pull.html)



## Webhook配置

参考[Webhook 实践 —— 自动部署](http://jerryzou.com/posts/webhook-practice/)

需要在腾讯云服务器上自动更新github pages的内容，于是采用webhook来实现。

```bash
npm install -g forever
forever statr server.js
```

## 显示 user.name 和 user.email

```bash
git config user.name
git config user.email
```

## 删除github的master分支

参考[删除github的master分支](http://blog.csdn.net/jefbai/article/details/44234383)

## Git push与pull的默认行为

参考[Git push与pull的默认行为](https://segmentfault.com/a/1190000002783245)

## GitHub 项目徽章的添加和设置

参考[GitHub 项目徽章的添加和设置](https://www.jianshu.com/p/e9ce56cb24ef)

以及

[shields.io](http://shields.io/)

## webhooks 响应特定分支的 push

参考[Web Hooks - execute only for specified branches #1176](https://github.com/gitlabhq/gitlabhq/issues/1176)@rtripault

提取 json 中的 `ref`，在 `do_POST()` 中进行设置响应动作。

## 命令行同步 fork 的仓库

1. 添加原仓库，比如

```bash
git remote add upstream git@github.com:LCTT/TranslateProject.git
```

查看当前远程仓库

```bash
git remote -v
```

2. pull

```bash
git pull upstream master
```

参考[Quick Tip: Sync a GitHub Fork via the Command Line](https://www.sitepoint.com/quick-tip-synch-a-github-fork-via-the-command-line/)

## 撤销未被 add 的文件

```bash
git checkout -- file
```

或者撤销所有更改

```bash
git checkout -- .
```

参考[git checkout all the files](https://stackoverflow.com/questions/29007821/git-checkout-all-the-files)

## rewrite 时百分比的含义

表示相似性。

参考[What does the message “rewrite … (90%)” after a git commit mean? [duplicate]](https://stackoverflow.com/questions/1046276/what-does-the-message-rewrite-90-after-a-git-commit-mean)

## Travis CI

中文折腾为什么 Travis CI 中用 '*' 号不会上传文件，最后还是指定了文件名。

刚刚找到解决方案 [How to deploy to github with file pattern on travis?](https://stackoverflow.com/questions/25929225/how-to-deploy-to-github-with-file-pattern-on-travis)

简单说要加上

```yml
file_glob: true
```

## 放弃本地修改

### 未 `git add`

直接用 `git checkout .`

### 已 `git add`

`git reset HEAD`

### 已 `commit`

`git reset --hard HEAD^`

参考[git 放弃本地修改](https://www.cnblogs.com/qufanblog/p/7606105.html)

## 列出另一分支的目录

```bash
git ls-tree master:dirname
```

## Github 回退

```bash
git reset --hard HEAD^
git push origin HEAD --force
```

参考
1. [github,退回之前的commit](https://www.cnblogs.com/xiaomengzhang/p/3240788.html)
2. [github 版本回退](https://blog.csdn.net/apple_wolf/article/details/53326187)

## cannot lock ref

参考 [git pull时遇到error: cannot lock ref 'xxx': ref xxx is at （一个commitID） but expected的解决办法](https://blog.csdn.net/qq_15437667/article/details/52479792)


另外试试 `git gc`

参考

1. [Git and nasty “error: cannot lock existing info/refs fatal](https://stackoverflow.com/questions/6656619/git-and-nasty-error-cannot-lock-existing-info-refs-fatal)

## remove untracked files

```
git clean -n
git clean -f
```

refer to [How to remove local (untracked) files from the current Git working tree](https://stackoverflow.com/questions/61212/how-to-remove-local-untracked-files-from-the-current-git-working-tree)

## 分支操作

详见 [Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v1/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6)

## `CRLF` & `LF`

双系统切换时，原先在 Win 处理，后来换到 Linux 处理，报出

```
CRLF will be replaced by LF.
```

设置 `autocrlf` 配置项：

- false 表示取消自动转换功能。适合纯 Windows
- true 表示提交代码时把 CRLF 转换成 LF，签出时 LF 转换成 CRLF。适合多平台协作
- input 表示提交时把 CRLF 转换成 LF，检出时不转换。适合纯 Linux 或 Mac

参考 [Git 多平台换行符问题(LF or CRLF)](https://blog.csdn.net/ljheee/article/details/82946368)

## Git push require username and password

possible reason: use the default HTTPS instead of SSH

correct this by

```
git remote set-url origin git@github.com:username/repo.git
```

参考 [Git push requires username and password](https://stackoverflow.com/questions/6565357/git-push-requires-username-and-password)


## manually resolve merge conflicts

1. open the file and edit
2. git add
3. git commit

refer to [How do I finish the merge after resolving my merge conflicts?](https://stackoverflow.com/questions/2474097/how-do-i-finish-the-merge-after-resolving-my-merge-conflicts)

## push 

采用低版本的 git， 如 `v1.8.3.1` 当 push 时会报出以下提醒信息，

```bash
warning: push.default is unset; its implicit value is changing in
Git 2.0 from 'matching' to 'simple'. To squelch this message
and maintain the current behavior after the default changes, use:

  git config --global push.default matching

To squelch this message and adopt the new behavior now, use:

  git config --global push.default simple

See 'git help config' and search for 'push.default' for further information.
(the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
'current' instead of 'simple' if you sometimes use older versions of Git)
```

而自己本机电脑一般采用 `v2.0+`， 比如当前 `v2.17.1` 的 git，所以直接采用 `simple` 模式便好了。