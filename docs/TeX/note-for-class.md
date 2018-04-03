1. [\newcommand vs \DeclareRobustCommand](https://tex.stackexchange.com/questions/61503/newcommand-vs-declarerobustcommand)
2. [increment-the-section-number](https://tex.stackexchange.com/questions/73104/increment-the-section-number-with-1-not-0-1)
3. [line-spacing](https://texblog.org/2011/09/30/quick-note-on-line-spacing/)

# 玄学问题

1. 标题加粗

```tex
\newcommand{\sanhao}{\fontsize{16pt}{\baselineskip}\selectfont\bfseries} 
\titleformat{\chapter}{\centering\sanhao}{\thechapter}{0em}{}
```

可以加粗

而

```tex
\newcommand{\sanhao}{\fontsize{16pt}{\baselineskip}\selectfont} 
\titleformat{\chapter}{\centering\sanhao\bfseries}{\thechapter}{0em}{}
```
不可以

参考 [Bold and italic subsection title with custom font size](https://tex.stackexchange.com/questions/165930/bold-and-italic-subsection-title-with-custom-font-size)