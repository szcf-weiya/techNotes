# Git Tips

## 教程

1. [阮一峰的网络日志](http://www.ruanyifeng.com/blog/2014/06/git_remote.html)
2. [这些GIT经验够你用一年了](https://zhuanlan.zhihu.com/p/22666153)
3. [Cheatsheet](git-cheat-sheet-education.pdf)

## 删除分支 

```bash
## Delete a remote branch
$ git push origin --delete <branch> # Git version 1.7.0 or newer 
$ git push origin :<branch> # Git versions older than 1.7.0

## Delete a local branch
$ git branch --delete <branch>
$ git branch -d <branch> # Shorter version
$ git branch -D <branch> # Force delete un-merged branches

## Delete a local remote-tracking branch
$ git branch --delete --remotes <remote>/<branch>
$ git branch -dr <remote>/<branch> # Shorter
$ git fetch <remote> --prune # Delete multiple obsolete tracking branches
$ git fetch <remote> -p # Shorter
```

adapted from [cmatskas/GitDeleteCommands.ps1](https://gist.github.com/cmatskas/454e3369e6963a1c8c89)

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

## 修改 origin

本地通过 git clone 得到仓库 https://github.com/CellProfiler/CellProfiler，后来需要在此基础上做些更改用到自己的项目中去，于是对原仓库进行了 fork，不过忘记了本地的仓库其实是从原仓库 clone 下来的，而非来自 fork 后的仓库，于是在 git push 会试图像原仓库进行 push，这当然是会被拒绝的。

简单的方法便是直接更改本地仓库的 origin，首先可以通过

```bash
git remote show origin
```

来查看当前 origin，前几行显示如下

```bash
$ git remote show origin 
* remote origin
  Fetch URL: git@github.com:CellProfiler/CellProfiler.git
  Push  URL: git@github.com:CellProfiler/CellProfiler.git
```

更改可以使用

```bash
git remote rm origin
git remote add origin git@github.com:szcf-weiya/CellProfiler.git
```

或者直接一步走

```bash
git remote set-url origin git@github.com:szcf-weiya/CellProfiler.git
```

注意到这里 url 都是直接取的 `git@github.com`，倘若用了 `https://github.com`，则 push 时会要求手动输入密码。

参考 [How to change the fork that a repository is linked to](https://stackoverflow.com/questions/11619593/how-to-change-the-fork-that-a-repository-is-linked-to)

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

## `git checkout`

- 撤销未被 add 的文件

```bash
git checkout -- file
```

- 撤销所有更改

```bash
git checkout -- .
```

参考[git checkout all the files](https://stackoverflow.com/questions/29007821/git-checkout-all-the-files)

- 从其他分支更新文件

```bash
git checkout master -- SOMEFILE
```

参考 [Quick tip: git-checkout specific files from another branch](http://nicolasgallagher.com/git-checkout-specific-files-from-another-branch/)

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

- 丢弃修改的内容：`git reset --hard HEAD^`
- 保留修改的内容：`git reset --soft HEAD^`
- 还有一种 `mixed`，也是默认不带参数的

参考

- [git 放弃本地修改](https://www.cnblogs.com/qufanblog/p/7606105.html)
- [github,退回之前的commit](https://www.cnblogs.com/xiaomengzhang/p/3240788.html)
- [github 版本回退](https://blog.csdn.net/apple_wolf/article/details/53326187)

### 放弃 reset

type

```bash
git reflog
```

to find the log of all ref updates, and then use

```bash
git reset 'HEAD@{1}'
```

to restore the corresponding state of `HEAD@{1}`, where `1` implies that last state, and it can be set as other refs.

refer to [How to undo 'git reset'?](https://stackoverflow.com/questions/2510276/how-to-undo-git-reset)

### 实际案例

某次写博客时，有一篇仍处于草稿箱的文件 A.md 暂时不想上传，但是更改了已经发表的文章 B.md 中的 typo，然后提交时一不留神直接用 

```bash
$ g "fix typo"
```

提交了，其中 `g` 是在 `.bashrc` 中自定义的一个函数，

```bash
# file .bashrc
# alias for git
func_g(){
    rm -f core
    git add .
    git commit -a -m "$1"
    git push
}
alias g=func_g
```

本来会直接 push 上去的，但是发现及时，Ctrl-C 取消掉了，不过 commit 需要回退。注意到这时并不想放弃修改的文档，所以 `--hard` 选项不可取，

```bash
$ git reset --soft HEAD^
```

这只是取消了 commit，但是仍然被 add 了，查看 git status，会有一行提示语，

```bash
  (use "git reset HEAD <file>..." to unstage)
```

所以下一步自然为

```bash
$ git reset HEAD A.md
```

有时可能删掉已经 add （甚至 commit，但还没 push） 的文件（比如下文中的 `Peek 2020-08-20 10-06.mp4`），首先如果 committed，则采用

```bash
$ git reset --soft HEAD^
```

取消 commit，然后 `git add` 就好了，但是继续撤销 add 的状态，并没有作用，而是提示

> Unstaged changes after reset:

```bash
$ git status
On branch master
Your branch is up to date with 'origin/master'.

Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        new file:   docs/Linux/Peek 2020-08-20 10-06.mp4

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        deleted:    docs/Linux/Peek 2020-08-20 10-06.mp4

$ git reset HEAD docs/Linux/Peek 2020-08-20 10-06.mp4
Unstaged changes after reset:
D       docs/Linux/Peek 2020-08-20 10-06.mp4
$ git add .
$ git status
On branch master
Your branch is up to date with 'origin/master'.
```

## 列出另一分支的目录

```bash
git ls-tree master:dirname
```

## 修改 commit 信息

If the latest commit， type

```bash
git commit --amend
```

to edit the commit message, and then

```bash
git push --force-with-lease
```

Refer to [Changing git commit message after push (given that no one pulled from remote)](https://stackoverflow.com/questions/8981194/changing-git-commit-message-after-push-given-that-no-one-pulled-from-remote).


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

## alias

本来想同时在一行命令中运行 `git add .` 和 `git commit -m`，但是如果采用 `;` 或者 `&&` 连接时都报错，

> error: unknown switch `m'

顺带发现了 `;` 和 `&&` 以及 `||` 的区别，详见 [Run multiple commands in one line with `;`, `&&` and `||` - Linux Tips](https://dev.to/0xbf/run-multiple-commands-in-one-line-with-and-linux-tips-5hgm)

- `;`: 无论第一个命令成功与否，都会运行第二个命令
- `&&`: 只有当第一个命令成功运行，才会运行第二个命令
- `||`: 只有当第一个命令失败后，才会运行第二个命令

后来发现这个，[Git add and commit in one command](https://stackoverflow.com/questions/4298960/git-add-and-commit-in-one-command)，可以通过 `git config` 来配置 alias，

> git config --global alias.add-commit '!git add -A && git commit'

则以后只需要调用

> git add-commit -m 'My commit message'

这跟在 `.bashrc` 中配置有异曲同工之妙！

## 标签

- 打标签

```bash
# annotated
git tag -a v1.0 -m "version 1"
# lightweight
git tag v1.0
# push to github
git push origin v1.0
# or 
git push origin --tags
```

详见 [2.6 Git 基础 - 打标签](https://git-scm.com/book/zh/v2/Git-%E5%9F%BA%E7%A1%80-%E6%89%93%E6%A0%87%E7%AD%BE)

- 删除标签

```bash
# local
git tag -d v1.0
# github
git push origin :refs/tags/v1.0
```

详见 [mobilemind/git-tag-delete-local-and-remote.sh](https://gist.github.com/mobilemind/7883996)

## color in git diff

在服务器上使用 `git diff`，没有像在本地 Ubuntu 那样用颜色高亮出不同的地方，这个可以通过在 `~/.gitconfig` 中进行配置，

```bash
[color]
  diff = auto
  status = auto
  branch = auto
  interactive = auto
  ui = true
  pager = true
```

参考 [How to colorize output of git?](https://unix.stackexchange.com/questions/44266/how-to-colorize-output-of-git)

顺带发现，如果在 markdown 中复制 git diff 的 output，可以使用 `diff` 的代码块，会将更改的部分高亮出来。

## rebase 放弃未 push 的 merges

现有本地、服务器仓库各一个，通过 GitHub 进行共享，原则上只通过本地编辑代码，然后在服务器中借助 `git pull` 进行更新代码。但是有一行代码由于文件路径原因，不得已在服务器端更改，在 `git pull` 之前进行了 `git add; git commit` 操作，所以 `pull` 的时候会产生 merge，而且这样下去每一次进行 `pull` 都会进行 merge。这样就很难看出服务器端真正更改的内容了，这时发现了 `git rebase`，这张图很好地展示了它要干的事情，

![](http://gitbook.liuhui998.com/assets/images/figure/rebase3.png)

这也正是我想要的，简单说就是把基于以前的更改转换成基于最新的更改。

然而我这情况还没这么直接，因为有过很多 merges，这些 merges 都想放弃掉。

```bash
$ git log --graph --format=%s
```

![](graph-merges.png)

后来发现了 [--rebase-merges](https://git-scm.com/docs/git-rebase/2.28.0#_rebasing_merges) 选项，似乎是在处理我的问题（~~最后还是理解反了~~），但是没高兴多久，服务器端的 git 版本只有 `1.8.3.1`，而这个选项至少要求 `2.x`。不过有注意到了 `--preserve-merges` 选项，新版本中已经 deprecated，而且文档中有这么一句，

```bash
[DEPRECATED: use --rebase-merges instead]
```

似乎也可以用，但是这个名字怎么感觉跟我的目标是相反的呢！?（此时还未意识到），后来简单试了一下

```bash
$ git rebase --preserve-merges
```

发现没啥变化。这才意识到自己理解错了这两个命令。

后来又注意到了 `--root` 选项，发现这个跟自己的目标挺像的，而且抱着搞坏了大不了重新 `git clone` 一下，所以果断去试了，history 确实都变成线性了

![](graph-linear.png)

但是似乎还没成功，commits 都变成不一样了，

![](rebase-root.png)

后来又瞎运行了下 

```bash
$ git rebase
```

发现竟然成功了，

![](rebase-again.png)

现在我甚至怀疑是不是一开始直接 `git rebase` 就好了，也不需要 `--root`，不过 `root` 是说可以 rebase 所有 reachable commits，没有什么 upstream 限制，不知道 merges 算不算 upstream 限制（**以后再玩玩**）. 如果需要更新的话，仍采用 `git pull`，但是 `git pull` 不是相当于 `git fetch & git merge` 么？！不过需要注意[加上 `--rebase` 选项](https://stackoverflow.com/a/36148845/8427014)

![](git-pull-with-rebase.png)

## 指定 commit 时间 

https://stackoverflow.com/questions/3895453/how-do-i-make-a-git-commit-in-the-past

## apply `.gitignore` to committed files

https://stackoverflow.com/questions/7527982/applying-gitignore-to-committed-files