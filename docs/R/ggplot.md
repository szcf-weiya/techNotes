# ggplot相关

## ggplot同时绘制多个密度图

参考[使用ggplot2同时绘制多个密度图](https://www.tuicool.com/articles/3aUnem7)

```r
plots <- NULL
for(i in colnames(train)){ 
    plots[[i]] <- ggplot(train) + 
    geom_density(aes_string(x = i, fill = 'is_black'), alpha = 0.5, show.legend = F) + 
    xlab("") + 
    ylab(""); 
}
plot_grid(plotlist = plots)
```

## ggplot绘制概率密度图

[ggplot2绘制概率密度图](http://www.cnblogs.com/wwxbi/p/6142410.html)

[Plotting distributions (ggplot2)](http://www.cookbook-r.com/Graphs/Plotting_distributions_(ggplot2)/)

## legend设置

参考[Legends (ggplot2)](http://www.cookbook-r.com/Graphs/Legends_(ggplot2)/)

### 默认情形

```r
library(ggplot2)
bp <- ggplot(data=PlantGrowth, aes(x=group, y=weight, fill=group)) + geom_boxplot()
bp
```

### 自定义图例的顺序

首先移除掉默认图例，有三种方式实现：
```r
# Remove legend for a particular aesthetic (fill)
bp + guides(fill=FALSE)

# It can also be done when specifying the scale
bp + scale_fill_discrete(guide=FALSE)

# This removes all legends
bp + theme(legend.position="none")
```

再改变默认顺序

```r
bp + scale_fill_discrete(breaks=c("trt1","ctrl","trt2"))
```

### 颠倒图例的顺序

```r
# These two methods are equivalent:
bp + guides(fill = guide_legend(reverse=TRUE))
bp + scale_fill_discrete(guide = guide_legend(reverse=TRUE))

# You can also modify the scale directly:
bp + scale_fill_discrete(breaks = rev(levels(PlantGrowth$group)))
```

### 隐藏图例标题

```r
# Remove title for fill legend
bp + guides(fill=guide_legend(title=NULL))

# Remove title for all legends
bp + theme(legend.title=element_blank())
```

### 自定义图例的标题及名称

两种方式，一种
另一种修改数据集

```r

```

### 图例的整体形状

```r
# Title appearance
bp + theme(legend.title = element_text(colour="blue", size=16, face="bold"))

# Label appearance
bp + theme(legend.text = element_text(colour="blue", size = 16, face = "bold"))
```

图例盒子

```r
bp + theme(legend.background = element_rect())
bp + theme(legend.background = element_rect(fill="gray90", size=.5, linetype="dotted"))
```

图例位置

```r
bp + theme(legend.position="top")

# Position legend in graph, where x,y is 0,0 (bottom left) to 1,1 (top right)
bp + theme(legend.position=c(.5, .5))

# Set the "anchoring point" of the legend (bottom-left is 0,0; top-right is 1,1)
# Put bottom-left corner of legend box in bottom-left corner of graph
bp + theme(legend.justification=c(0,0), legend.position=c(0,0))

# Put bottom-right corner of legend box in bottom-right corner of graph
bp + theme(legend.justification=c(1,0), legend.position=c(1,0))
```

### 隐藏图例的slashes

```r
# No outline
ggplot(data=PlantGrowth, aes(x=group, fill=group)) +
    geom_bar()

# Add outline, but slashes appear in legend
ggplot(data=PlantGrowth, aes(x=group, fill=group)) +
    geom_bar(colour="black")

# A hack to hide the slashes: first graph the bars with no outline and add the legend,
# then graph the bars again with outline, but with a blank legend.
ggplot(data=PlantGrowth, aes(x=group, fill=group)) +
    geom_bar() +
    geom_bar(colour="black", show.legend=FALSE)
```