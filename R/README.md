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

## shiny

[init](https://github.com/rstudio/shiny-server/issues/153)

## 删除
```
rm(list = ls(all = TRUE))
```

## interval
http://blog.sciencenet.cn/blog-54276-288414.html

## window 安装包
切换到R的安装路径下，在etc文件夹中编辑文件Rprofile.site文件

```
# set a CRAN mirror
    local({r <- getOption("repos")
		r["CRAN"] <- "http://mirrors.ustc.edu.cn/CRAN/"
          options(repos=r)}) 
```

## sort(), rank(), order()

http://blog.sina.com.cn/s/blog_6caea8bf0100spe9.html

sort(x)是对向量x进行排序，返回值排序后的数值向量。rank()是求秩的函数，它的返回值是这个向量中对应元素的“排名”。而order()的返回值是对应“排名”的元素所在向量中的位置。

![](sro.png)

## Interpreting Residual and Null Deviance in GLM R

![](glm.png)

Refer to https://stats.stackexchange.com/questions/108995/interpreting-residual-and-null-deviance-in-glm-r


