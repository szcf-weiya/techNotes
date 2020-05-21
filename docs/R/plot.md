# 普通作图

## 光滑曲线

```r
x <- 1:10
y <- c(2,4,6,8,7,12,14,16,18,20)
lo <- loess(y~x)
plot(x,y)
lines(predict(lo), col='red', lwd=2)
```

参考[How to fit a smooth curve to my data in R?](https://stackoverflow.com/questions/3480388/how-to-fit-a-smooth-curve-to-my-data-in-r)

## Low-Level Graphics

Suppose I want to draw the following figure with R,

![](low-level-plot.PNG)

At first, I try to use `xaxt` option to remove the axis, but the box still remains, just same as the [one question in the StackOverflow](https://stackoverflow.com/questions/4785657/how-to-draw-an-empty-plot), and I found a possible solution, directly use

```r
plot.new()
```

All is well before I tried to add the text $\rho$, if I use

```r
text(0.8, 0, expression(rho), cex = 2)
```

it is OK, but it is exactly on the axis, not proper, but when I tried smaller $y$-coordinate, such as -0.1, the text cannot appear, which seems out of the figure. I have tried `par()` parameters, such as `mar`, but does not work.

Then I have no idea, and do not know how to google it. And even though I want to post an issue in the StackOverflow. But a [random reference](https://www.stat.auckland.ac.nz/~ihaka/120/Notes/ch03.pdf) give me ideas, in which the example saves me,

```r
> plot.new()
> plot.window(xlim=c(0,1), ylim=c(5,10))
> abline(a=6, b=3)
> axis(1)
> axis(2)
> title(main="The Overall Title")
> title(xlab="An x-axis label")
> title(ylab="A y-axis label")
> box()
```

Then I realized that I should add

```r
plot.window(xlim = c(0, 1), ylim = c(-0.1, 0.9))
```

## 画图间距

有时通过 `par(mfrow=c(2,1))` 画图时间距过大，这可以通过 `mar` 来调节，注意到

- `mar` 调节单张图的 margin
- `oma` 调节整张图外部的 margin

参考 [how to reduce space gap between multiple graphs in R](https://stackoverflow.com/questions/15848942/how-to-reduce-space-gap-between-multiple-graphs-in-r)

比如，[B spline in R, C++ and Python](https://github.com/szcf-weiya/ESL-CN/commit/a79daf246320a7cd0ae57c0b229fc096d98483f6)

## pairs中自定义panel函数

问题来自[R语言绘图？ - 知乎](https://www.zhihu.com/question/268216627/answer/334393347)

![](https://pic4.zhimg.com/v2-ad40c8c5a577f9ed3ae82dd43c7dfdcf_r.jpg)

```r
my.lower <- function(x,y,...){
  points(x, y)
  lines(lowess(x, y), col = "red", lwd=2)
}

my.upper <- function(x, y, ...){
  cor.val = round(cor(x,y), digits = 3)
  if (abs(cor.val) > 0.5){
    text(mean(x), mean(y), cor.val, cex = 3)
    text(sort(x)[length(x)*0.8], max(y), '***', cex = 4, col = "red")
  } else
  {
    text(mean(x), mean(y), cor.val, cex = 1)
  }
}

pairs(iris[1:4], lower.panel =my.lower, upper.panel = my.upper)
```

参考 [Different data in upper and lower panel of scatterplot matrix](https://stackoverflow.com/questions/15625510/different-data-in-upper-and-lower-panel-of-scatterplot-matrix)
