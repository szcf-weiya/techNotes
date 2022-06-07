# XeLaTeX on Travis CI

参考

1. [Setup LaTeX PDF build using Travis CI](https://hv.pizza/blog/setup-latex-pdf-build-using-travis-ci/)
2. [Document building & versioning with TeX document, Git, Continuous Integration & Dropbox](https://hv.pizza/blog/document-building-versioning-with-tex-document-git-continuous-integration-dropbox/)

## first example

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

## second example

```yml
before_install:
  - "sudo apt-get update && sudo apt-get install --no-install-recommends texlive-fonts-recommended texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-xetex dvipng"

script:
  - cd proposal
  - xelatex test
```

succeed!

## install Chinese fonts

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

## use xeCJK

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

```bash
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

## test main.tex

称缺失 `xcolor.sty`，而用 `apt-cache search xcolor | grep tex` 得到的包为 extra 和 recommended，但这两个包已经安装了，所以怀疑是 `--no-install-recommends`。

不过突然想到，版本不对，我本机为 TeXlive 2015，而 Travis CI 上为 2013，所以我又在服务器上进行测试，发现 `xcolor` 在 `latex-xcolor` 包中。
