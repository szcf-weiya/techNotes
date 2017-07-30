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


# 调整目录的显示层数

在使用tableofcontents 命令的时候，可分别显示chapter ， section ，subsection ，subsubsection 等目录，有时候，不希望显示级别较低的内容，比如只显示到chapter 和section，而subsection 和subsubsection 不显示，这时候可通过命令setcounter 命令来控制，具体做法如下：

```
\setcounter{tocdepth}{2}
```
即只显示两级目录。

# Add Metadata to XeTeX PDF's

```
\usepackage[pdfauthor={Your Name},
            pdftitle={The Title},
            pdfsubject={The Subject},
            pdfkeywords={Some Keywords},
            pdfproducer={XeLateX with hyperref},
            pdfcreator={Xelatex}]{hyperref}
```

# 希腊字母加粗问题
[reference](http://blog.sina.com.cn/s/blog_5e16f1770100ks8l.html)
方案一、用\usepackage{amsmath}
\boldsymbol{\sigma}

\mathbf 只对公式中的普通字母ABC...abcdef等起作用。

方案二、更好的方法是使用\usepackage{bm}
\bm{}来加粗。

# 数学字体加粗

使用mathbf加粗完后斜体不见了，这不是想要的结果

[LaTeX数学字体加粗问题](http://blog.sina.com.cn/s/blog_5e16f1770100nqwx.html)

