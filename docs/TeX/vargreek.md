# Variant forms of Greek symbols

## Attempt 1: `MinionPro`

```tex
\documentclass{article}
\usepackage{MinionPro} %! FAIL TO INSTALL THE PACKAGE
\begin{document}
	$\epsilon\varepsilon\phi\varphi\theta\vartheta\kappa\varkappa\beta\varbeta\pi\varpi\rho\varrho$
\end{document}
```

首先在 T460p 上进行了尝试，不过直接运行会报出 

> ! LaTeX Error: File `MinionPro.sty` not found.

```bash
$ locate MinionPro.sty
$ apt-cache search MinionPro
texlive-fonts-extra - TeX Live: Additional fonts
```

于是试图装这个包，但是因为提示说要占用 1300+ MB 的空间。因为心疼硬盘大小，所以转而在 G40 上测试，

```bash
$ sudo apt install texlive-fonts-extra
Reading package lists... Done
Building dependency tree       
Reading state information... Done
texlive-fonts-extra is already the newest version (2019.202000218-1).
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
```

但是显示已经装好的，然而确实编译过程中找不到这个包。甚至试图重装也是没有这个包，

```bash
$ sudo apt reinstall texlive-fonts-extra
Reading package lists... Done
Building dependency tree       
Reading state information... Done
0 upgraded, 0 newly installed, 1 reinstalled, 0 to remove and 0 not upgraded.
Need to get 460 MB of archives.
After this operation, 0 B of additional disk space will be used.
Get:1 http://cn.archive.ubuntu.com/ubuntu focal/universe amd64 texlive-fonts-extra all 2019.202000218-1 [460 MB]
Fetched 460 MB in 1min 30s (5,123 kB/s)                                                                                                                              
(Reading database ... 619949 files and directories currently installed.)
Preparing to unpack .../texlive-fonts-extra_2019.202000218-1_all.deb ...
Unpacking texlive-fonts-extra (2019.202000218-1) over (2019.202000218-1) ...
Setting up texlive-fonts-extra (2019.202000218-1) ...
Processing triggers for tex-common (6.13) ...
Running mktexlsr. This may take some time... done.
Running mtxrun --generate. This may take some time... done.
Running updmap-sys. This may take some time... done.
Running mktexlsr /var/lib/texmf ... done.
$ locate MinionPro.sty

```

另外根据[安装指示](https://ctan.org/tex-archive/fonts/minionpro)，

```none
  $ otfinfo -v MinionPro-Regular.otf
  Version 2.012;PS 002.000;Core 1.0.38;makeotf.lib1.6.6565

2) Copy your OpenType font files into the otf directory.

  $ cp /some/path/*.otf otf
```

似乎需要预装 `MinionPro-Regular.otf` 字体，然而根据 [Adobe fonts](https://fonts.adobe.com/fonts/minion) 给的信息，该字体是 "Available with CC"，也就是说需要 [Adobe Creative Cloud subscription](https://community.adobe.com/t5/adobe-fonts/how-to-use-available-with-cc-font-on-a-website/m-p/10444242). 然而似乎学校没有订购，办公室电脑也没有找到该字体。

## Attempt 2: `unicode-math`

```tex
\documentclass{article}
\usepackage{unicode-math}
\begin{document}
    Asana Math:
	\setmathfont{Asana Math}
	$\symbol{"003D0}$
\end{document}
```

需要通过 `\symbol` 来使用该符号，不够便捷。而且如果使用其它的字符，还要预先知道其 unicode 码

## Attempt 3: `mathspec`

```tex
\documentclass{article}
\usepackage{mathspec}
\setmathfont(Greek){Asana-Math}
\begin{document}
    $\epsilon\varepsilon\phi\varphi\theta\vartheta\kappa\varkappa\beta\varbeta\pi\varpi\rho\varrho$
\end{document}
```

这个包的文档中有列出可用的 variant forms，不过 `\epsilon` 和 `\varepsilon` 跟我之前的认知反过来了。

![image](https://user-images.githubusercontent.com/13688320/118352489-59202c00-b594-11eb-89e4-c9695617fb82.png)

一开始便确定了 `Asana Math` 能够很好地支持该字体，而且 `FreeMono` 也可以，只不过字体有点差异。

## Run in Batch

为了方便比较不同字体的效果，首先尝试在正文中切换

```tex
\setmathfont(Greek){Asana-Math}
```

但是报错提示说只能在 preamble 中定义。不过注意到如果不加 `mathspec` package，在文内 `\setmathfont{...}` 是可以的（注意此时没有指定 Greek）。

既然无法同时在一篇文档里面展现，那就多篇呗。因为 `article` 默认 A4 paper，也看过 `geometry`，似乎也只提供常见的 paper 大小，不能像 `standalone` 环境那样，然而直接使用 `standalone` 又不可行，似乎它只适用于 tikz。于是便想到 crop pdf。

这篇回答[Command line tool to crop PDF files](https://askubuntu.com/questions/124692/command-line-tool-to-crop-pdf-files) 提到， latex 本身有提供命令 `pdfcrop`，查看其帮助文档不能用 `man`，而是用 `texdoc`，从文档本身了解到 `--help` 可以看帮助文档。参数 `--bbox` 可以指定裁剪区域，而这个参数定义又是来自于 ghostscript 的，所以预先需要了解[这些参数是怎么定义的](https://stackoverflow.com/questions/6250064/ghostscript-boundingbox-values)，

```bash
$ gs -sDEVICE=bbox vargreek.pdf 
GPL Ghostscript 9.26 (2018-11-20)
Copyright (C) 2018 Artifex Software, Inc.  All rights reserved.
This software comes with NO WARRANTY: see the file PUBLIC for details.
Processing pages 1 through 1.
Page 1
%%BoundingBox: 149 139 308 714
%%HiResBoundingBox: 149.867995 139.247996 307.295991 713.465978
```

其中 `--box <left> <bottom> <right> <top>` 四个参数的含义（72 个 points 为 1inch）是

- `<left> <bottom>`: 左下角，距离左边边界 `<left>` 个 points，而距离底边边界 `<bottom>` 个 points
- `<right> <top>`: 右上角，距离左边边界 `<right>` 个 points，而距离底边边界 `<top>` 个 points 

然后开始写 Makefile，通过 `jobname` 指定字体名，这样每次添加一个字体都要新增一个 object，不够智能，想通过写 for 循环来实现。不过在 Makefile 写循环不是很熟悉，索性直接写 shell 脚本。

首先便需要通过 `fc-list` 提取字体名，不过这里需要注意，字体名是用空格分隔的，所以需要加上引号，但是这还不够，如果采用

```bash
for font in ${fonts[@]}; do
	$font
```

`$font` 仍会以空格进行分隔，进而只提取第一个元素。需要换成

```bash
for ((i = 0; i < ${#fonts[@]}; i++)); do
    font=${fonts[$i]}
```

参考 [Bash array with spaces in elements](https://stackoverflow.com/questions/9084257/bash-array-with-spaces-in-elements)

但是如果用 `fc-list` 提取的话，不如直接将空格换成 `-`，后面需要用到原形式时再换过来。

```bash
# orig=${font//[-]/ }
# echo $font ${font//[-]/ }
```

这个技巧参考[Replacing some characters in a string with another character](https://stackoverflow.com/questions/2871181/replacing-some-characters-in-a-string-with-another-character)

默认情况下，latex 如果出错，比如找不到字体文件，为了避免这个问题，添加参数 `-halt-on-error` 遇到错误即退出，类似参数的选择解释详见 [What is the difference between “-interaction=nonstopmode” and “-halt-on-error”?](https://tex.stackexchange.com/questions/258814/what-is-the-difference-between-interaction-nonstopmode-and-halt-on-error)

最后，再将这些文档通过 `pdftk` 结合在一起。

最终成功的版本已整理至[个人模板仓库](https://github.com/szcf-weiya/TeXtemplates/tree/515c50cf13abd3231c762701f2c3d0baaa4d7e11/_includes/vargreek)。

## References 

- [Use the beta variant](https://tex.stackexchange.com/questions/191759/use-the-beta-variant)
