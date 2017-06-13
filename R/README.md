# R notes

```R
for (i in c(1:n-1))
  print(i)
#0
#1
#2
for (i in c(1:(n-1)))
  print(i)
#1
#2
```

## Error in plot.new() : figure margins too large in R

[reference](http://stackoverflow.com/questions/12766166/error-in-plot-new-figure-margins-too-large-in-r)

![](error_too_large_for_figure.png)


## linux更新的问题

查看https://mirrors.tuna.tsinghua.edu.cn/CRAN/ 中的README.md文件

## 更新记录
1. 找到源码文件夹
```{r}
sudo make uninstall
```
2. 配置source.list，安装。

## 终端执行R code

[reference](https://stackoverflow.com/questions/18306362/run-r-script-from-command-line)
