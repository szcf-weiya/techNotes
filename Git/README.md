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

[](http://www.cnblogs.com/lout/p/6111739.html)

