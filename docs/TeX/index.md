# 关于 TeX

## 缺少字体

报错信息如下图

![](font_error.png)

解决方案为
```bash
sudo apt-get install texlive-fonts-extra
```

有时需要考虑
```
sudo apt-get install texlive-fonts-recommand
```

## 安装 TeXLive 2016 
1. [How to install “vanilla” TeXLive on Debian or Ubuntu?](http://tex.stackexchange.com/questions/1092/how-to-install-vanilla-texlive-on-debian-or-ubuntu/95373#95373)
2. [Linux 系统下原版 texlive 2016 的安装与配置](http://www.cnblogs.com/wenbosheng/archive/2016/08/03/5725834.html)

记住勾选 create symlinks to standard directories

## 缺少 `.sty` 文件

比如缺少 `mhchem.sty`

```error
! LaTeX Error: File 'mhchem.sty' not found
```
解决方案为
```bash
# 1. 检查是否存在 `mhchem.sty` 文件
$ locate mhchem.sty
# 2. 查找需要安装的 package
$ apt-cache search mhchem | grep tex
# texlive-science - TeX Live: Natural and computer sciences
# 3. 安装相应的package
$ sudo apt-get install texlive-science
```

参考 [Latex can't find .sty files altough packages are installed — TexLive, Ubuntu 12.04](http://tex.stackexchange.com/questions/158700/latex-cant-find-sty-files-altough-packages-are-installed-texlive-ubuntu-12)


## Biblatex

记得设置 texstudio 的 biblatex 编译方式，设为 biber，默认为 bibtex.


## 调整目录的显示层数

在使用 `tableofcontents` 命令的时候，可分别显示 `chapter`，`section`，`subsection`，`subsubsection` 等目录，有时候，不希望显示级别较低的内容，比如只显示到 `chapter` 和 `section`，而 `subsection` 和 `subsubsection` 不显示，这时候可通过命令 `setcounter` 命令来控制，具体做法如下：

```tex
\setcounter{tocdepth}{2}
```

即只显示两级目录。

## pdf 添加 Metadata 信息

```tex
\usepackage[pdfauthor={Your Name},
            pdftitle={The Title},
            pdfsubject={The Subject},
            pdfkeywords={Some Keywords},
            pdfproducer={XeLateX with hyperref},
            pdfcreator={Xelatex}]{hyperref}
```

## 希腊字母加粗问题

`\mathbf` 只对公式中的普通字母 `ABC...abcdef` 等起作用。

采用

```tex
\usepackage{amsmath}
\boldsymbol{\sigma}
```

或

```tex
\usepackage{bm}
\bm{\sigma}
```

参考 [LaTeX技巧326：希腊字母的加粗问题](http://blog.sina.com.cn/s/blog_5e16f1770100ks8l.html)


## 数学字体加粗

使用 `\mathbf` 加粗完后斜体不见了，这不是想要的结果

```tex
\usepackage{amsmath}
\boldmath
$$
f(x,y) = 3(x+y)y / (2xy-7)
$$
\unboldmath
```

或

```tex
\usepackage{bm}
$$
\bm{f(x,y) = 3(x+y)y / (2xy-7)}
$$
```

参考 [LaTeX 数学字体加粗问题](http://blog.sina.com.cn/s/blog_5e16f1770100nqwx.html)


## “texi2dvi” command not found

```bash
sudo apt-get install texinfo
```

## 数学公式插入图片

参考 [Can I insert an image into an equation?](https://tex.stackexchange.com/questions/11069/can-i-insert-an-image-into-an-equation)

## beamer 中 frame 的 fragile 选项

参考 [LaTeX 技巧 573：beamer 中使用 Listings 包出现的错误](http://blog.sina.com.cn/s/blog_5e16f1770102dxps.html)

错误描述

```none
Runaway argument?
! Paragraph ended before \lst@next was complete.
<to be read again>
                   \par
l.68 \end{frame}
?
```

解决方案

```tex
\begin{frame}[fragile]
\frametitle{Your title}

\begin{lstlisting}
code
\end{lstlisting}
\end{frame}
```

## beamer 中数学字体

默认数学字体挺丑的，可以在导言区加入

```tex
\usefonttheme[onlymath]{serif}
```

下面摘录自 [Beamer 中数学符号字体 ](http://blog.sina.com.cn/s/blog_4b91d3b50101lupb.html)

> 关于tex的字体样式，其实是通用的，与css和windows字体等，都是通用的。来源于西方的字母写法，大致可分为两类：serif （衬线）和sans-serif（无衬线）。

> 所谓衬线是字体的末端加强，便于阅读。如通常见的Times New Roman, 宋体。sans-serif（sans 源自法语，表示“没有”）字体的代表如Arial，隶书，幼圆。由于衬线的强化作用，serif字体作为正文具有易读性。因此存在大段文本的情况下，常使用衬线字体。但做幻灯片的话，衬线字体会因字体粗细不同，反倒可能降低辨识度。因此建议标题用衬线字体，正文用非衬线字体。

> 数学符号用衬线字体相对美观一些，而Beamer如果不另行设置，默认全文使用sans-serif字体。因此按上述方式设置一下即可。

## 源码安装 texlive

[How to install “vanilla” TeXLive on Debian or Ubuntu?](https://tex.stackexchange.com/questions/1092/how-to-install-vanilla-texlive-on-debian-or-ubuntu/95373#95373)

以及
[How to properly install and use texlive with package manager in 14.04](https://askubuntu.com/questions/485514/how-to-properly-install-and-use-texlive-with-package-manager-in-14-04)

## beamer 中导入视频

[Can XeLaTeX | LuaTeX import movies?](https://tex.stackexchange.com/questions/12790/can-xelatex-luatex-import-movies)


## LaTeX 中的定理环境

[LaTeX中的定理环境](http://blog.sina.com.cn/s/blog_62b52e290100yifl.html)


## makeatletter and makeatother

参考 [What do \makeatletter and \makeatother do?](https://tex.stackexchange.com/questions/8351/what-do-makeatletter-and-makeatother-do)

## Why can't the end code of an environment contain an argument?

参考[Why can't the end code of an environment contain an argument?](https://tex.stackexchange.com/questions/17036/why-cant-the-end-code-of-an-environment-contain-an-argument)

## What are category codes?

参考[What are category codes?](https://tex.stackexchange.com/questions/16410/what-are-category-codes)

## When to use @ in an \if statement

参考[When to use @ in an \if statement](https://tex.stackexchange.com/questions/27803/when-to-use-in-an-if-statement)

## Formatting section titles

参考[Formatting section titles](https://tex.stackexchange.com/questions/36609/formatting-section-titles)

## Letex 画复杂表格的方法

参考[Letex画复杂表格的方法](http://blog.csdn.net/jiakunboy/article/details/46355951)


## latex 列举 enumerate 编号 样式设定

参考[latex 列举 enumerate 编号 样式设定](http://blog.sina.com.cn/s/blog_7983e5f101019wwq.html)

## 二阶导数符号

在tex中一般直接用

```tex
$f''$
```

但是在md中，当渲染成网页后，有时会渲染成了普通的引号，如下图

![](prime.PNG)

参考[How to write doubleprime in latex](https://tex.stackexchange.com/questions/210290/how-to-write-doubleprime-in-latex)

1. `\dprime`和`\trprime`需要`unicode-math`
2. `f^{\prime\prime}`代替`f''`可以解决问题。

## bibtex文献加颜色


两种方式

第一种

```tex
\hypersetup{
    colorlinks,
    citecolor=green,
    linkcolor=black
}
```

但这个只会对参考文献中可点击的部分起作用，比如实际中只对年份起了作用。

第二种可以自定义命令

```tex
\DeclareCiteCommand{\cite}
  {\color{red}\usebibmacro{prenote}}%
  {\usebibmacro{citeindex}%
   \usebibmacro{cite}}
  {\multicitedelim}
  {\usebibmacro{postnote}}

\DeclareCiteCommand{\parencite}[\mkcolorbibparens]
  {\usebibmacro{prenote}}%
  {\usebibmacro{citeindex}%
   \usebibmacro{cite}}
  {\multicitedelim}
  {\usebibmacro{postnote}}
```


参考
1. [Beamer, Citation coloring](https://tex.stackexchange.com/questions/369710/beamer-citation-coloring)

## 将引用的年份用括号框起来

参考[Put parentheses around year in citation](https://tex.stackexchange.com/questions/104518/put-parentheses-around-year-in-citation)

采用`natbib`中的`\citet`

但若已经按照上个问题设置了颜色，则颜色失效。

## 设置item之间的间隔

直接用`itemsep`命令，如

```tex
\begin{itemize}
  \setlength\itemsep{1em}
  \item one
  \item two
  \item three
\end{itemize}
```

## xelatex pdf on Travis CI

参考 

1. [Setup LaTeX PDF build using Travis CI](https://hv.pizza/blog/setup-latex-pdf-build-using-travis-ci/)
2. [Document building & versioning with TeX document, Git, Continuous Integration & Dropbox](https://hv.pizza/blog/document-building-versioning-with-tex-document-git-continuous-integration-dropbox/)

### first example

- no chinese
- no other packages

```yml
before_install:
  - "sudo apt-get update && sudo apt-get install --no-install-recommends texlive-fonts-recommended texlive-latex-extra texlive-fonts-extra texlive-latex-recommended dvipng"

script:
  - cd proposal
  - xelatex test
```

It turns out that no xelatex.

### second example
```yml
before_install:
  - "sudo apt-get update && sudo apt-get install --no-install-recommends texlive-fonts-recommended texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-xetex dvipng"

script:
  - cd proposal
  - xelatex test
```

succeed!

### install Chinese fonts

```yml
sudo: required
dist: trusty
before_install:
  - "sudo apt-get update && sudo apt-get install --no-install-recommends texlive-fonts-recommended texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-xetex dvipng"
  - "wget -c https://sourceforge.net/projects/zjuthesis/files/fonts.tar.gz/download && tar xzf fonts.tar.gz && sudo mkdir -p /usr/share/fonts/truetype/custom/ && mv fonts/* /usr/share/fonts/truetype/custom/ && sudo fc-cache -f -v"

script:
  - cd proposal
  - xelatex test
```

wrong error in wget. It should be 

```yml
wget -c https://sourceforge.net/projects/zjuthesis/files/fonts.tar.gz/download -O fonts.tar.gz
```

and mv should add `sudo`

```yml
sudo mv fonts/* /usr/share/fonts/truetype/custom/ 
```

succeed!

### use xeCJK

```yml
# test.tex
\documentclass{article}
\usepackage{xeCJK}
\setCJKmainfont{STFANGSO.TTF}
\begin{document}
test TeX on Travis CI via xelatex.
毕业论文（设计）题目 

浙江大学本科生毕业论文（设计）
\end{document}

# .travis.yml
sudo: required
dist: trusty
before_install:
  - "sudo apt-get update && sudo apt-get install --no-install-recommends texlive-fonts-recommended texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-xetex dvipng"
  - "wget -c https://sourceforge.net/projects/zjuthesis/files/fonts.tar.gz/download -O fonts.tar.gz && tar xzf fonts.tar.gz && sudo mkdir -p /usr/share/fonts/truetype/custom/ && sudo mv fonts/* /usr/share/fonts/truetype/custom/ && sudo fc-cache -f -v"

script:
  - cd proposal
  - xelatex test
```

It reports that

```error
! Font EU1/lmr/m/n/10=[lmroman10-regular]:mapping=tex-text at 10.0pt not loadab
le: Metric (TFM) file or installed font not found.
```

refer to [! Font EU1/lmr/m/n/10=[lmroman10-regular]:mapping=tex-text at 10.0pt not loadab le: Metric (TFM) file not found](https://tex.stackexchange.com/questions/129799/font-eu1-lmr-m-n-10-lmroman10-regularmapping-tex-text-at-10-0pt-not-loadab)

try to install `fonts-lmodern` fist.

```yml
sudo apt-get install fonts-lmodern
```

succeed!

However, I cannot find `test.pdf` in releases, and it contains other files. So set

```yml
skip_cleanup: true
```

and fix typo from `tages` to `tags`.

But it still cannot find the `test.pdf`, and even cannot find any release

so I change `file: proposal/test.pdf` to `file: test.pdf`.

It reports that `Skipping a deployment with the releases provider because this is not a tagged commit`

It still failed.

Then use the normal method

```
git add .
git commit -m "    "
git tag "test"
git push origin master --tags
```

It succeed!

Then I found that if use `before_deploy`, it seems that no need to set `on.tags = true`.

### test main.tex

称缺失 `xcolor.sty`，而用 `apt-cache search xcolor | grep tex` 得到的包为 extra 和 recommended，但这两个包已经安装了，所以怀疑是 `--no-install-recommends`。

不过突然想到，版本不对，我本机为 TeXlive 2015，而 Travis CI 上为 2013，所以我又在服务器上进行测试，发现 `xcolor` 在 `latex-xcolor` 包中。

## beamer 不兼容 enumitem

详见[lists - Trouble combining enumitem and beamer - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/31505/trouble-combining-enumitem-and-beamer)

摘录其中一句话

> enumitem “disturbs” beamer. 

## 替换换行符

详见[strings - Replacing substrings by linebreaks - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/178610/replacing-substrings-by-linebreaks)

这是 zju-thesis 模板允许封面标题换行的处理策略，同时参见 

1. [Line break inside \makebox](https://tug.org/pipermail/texhax/2007-November/009566.html)

## APA 带编号

默认 APA 格式的参考文献是不带标号的，如果需要，参考

[APA bibliography style with numbers](https://tex.stackexchange.com/questions/373336/apa-bibliography-style-with-numbers)

## itemize 环境下不同 item 中数学公式对齐

参考[Sharing alignment between equations in two different items](https://tex.stackexchange.com/questions/29119/sharing-alignment-between-equations-in-two-different-items)

## 数学公式后标点

参考[For formal articles, should a displayed equation be followed by a punctuation to conform to the language grammar ?](https://tex.stackexchange.com/questions/7542/for-formal-articles-should-a-displayed-equation-be-followed-by-a-punctuation-to)

## Biblatex/biber fails with a strange error about missing recode_data.xml file

参考[Biblatex/biber fails with a strange error about missing recode_data.xml file](https://tex.stackexchange.com/questions/140814/biblatex-biber-fails-with-a-strange-error-about-missing-recode-data-xml-file/258217#258217)

## 同时在等号上下注明

参考 [Writing above and below a symbol simultaneously ](https://tex.stackexchange.com/questions/123219/writing-above-and-below-a-symbol-simultaneously)

## remark environment

In beamer,  need to use
```tex
\newtheorem{remark}{Remark} 
```
before.

refer to [Theorem Numbering in beamer](https://tex.stackexchange.com/questions/188379/theorem-numbering-in-beamer?rq=1)