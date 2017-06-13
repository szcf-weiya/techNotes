# Git Tips

## 删除远程分支

```git
git branch -r -d origin/branch-name
git push origin :branch-name
```

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
