# Git Tips

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
git clone -a
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

## git branch
[ref](http://blog.csdn.net/guang11cheng/article/details/37757201)


## git init


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

## 教程

1. [阮一峰的网络日志](http://www.ruanyifeng.com/blog/2014/06/git_remote.html)
