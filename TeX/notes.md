# 问题解决

## 关于字体
![](font_error.png)

```bash
sudo apt-get install texlive-fonts-extra
```
## TeXLive2016安装
1. [tex.stackexchange.com](http://tex.stackexchange.com/questions/1092/how-to-install-vanilla-texlive-on-debian-or-ubuntu/95373#95373)
2. [www.cnblogs.com](http://www.cnblogs.com/wenbosheng/archive/2016/08/03/5725834.html)

记住勾选create symlinks to standard directories

## Install mhchem.sty

参考[tex.stackexchange](http://tex.stackexchange.com/questions/158700/latex-cant-find-sty-files-altough-packages-are-installed-texlive-ubuntu-12)

```error
! LaTeX Error: File 'mhchem.sty' not found
```

### Step 1: 检查是否存在mhchem.sty文件

```bash
$ locate mhchem.sty
```

### Step 2: 查找需要安装的package

```
$ apt-cache search mhchem | grep tex
```

得到
```bash
texlive-science - TeX Live: Natural and computer sciences
```

于是

### Step 3: 安装相应的package

```
sudo apt-get install texlive-science
```

## Biblatex

记得设置texstudio的biblatex编译方式，设为biber，默认为bibtex.
