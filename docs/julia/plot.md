# 作图

## 等高线图 (contour)

```julia
using Plots
using Distributions

x_grid = range(-2, 2, length=100)
y_grid = range(-2, 2, length=100)
Sigma = [1 0.9; 0.9 1];
contour(x_grid, y_grid, (x, y)->pdf(MvNormal([0, 0], Sigma), [x, y]), cbar=false)
```

## 网格 3D 图

要求颜色随着高度渐变，这包含两部分的渐变，

- 不同纬度上的每一个截面圆指定颜色
- 经线的的不同维度需要分段指定颜色

![](https://user-images.githubusercontent.com/13688320/83348932-2bca7600-a363-11ea-9ad0-07da29102ca0.png)

第一点其实很简单，当生成好一组渐变颜色后，比如

```julia
colors = cgrad(:summer, nz, categorical = true)
```

更多的选择详见 [Colorschemes](https://docs.juliaplots.org/latest/generated/colorschemes/)

在循环画图时，每次指定一种颜色便 OK 了。

第二点其实也很简单，在画一条曲线时，如果 linecolor 指定为颜色**列**向量，比如 `[:red, :green, :orange]`，则每一段的颜色会循环采用该列向量中的颜色，则当列向量长度刚好等于区间段的个数，则每一段都会以不同的颜色绘制。需要注意到是，颜色**行**向量用于画多条曲线时为每一条曲线指定不同的颜色。

但最后第二点折腾了有点久，直接把 `colors` 带进去并不行，后来才发现它不是简单的颜色列向量，里面还包含其它信息，最后采用 `colors.colors.colors` 才成功。详见 [ESL-CN/code/SOM/wiremesh.jl](https://github.com/szcf-weiya/ESL-CN/blob/5e8d95299d4c53d8f509324546b40c65b31a3666/code/SOM/wiremesh.jl#L52-L54)

另外，也尝试过[官方文档例子](http://docs.juliaplots.org/latest/generated/gr/#gr-ref24-1)中 `zcolor` 参数，但是似乎只针对 marker 起作用，当然是跟 `m = ` 参数配合的效果，第一个元素代表大小，第二个透明度。所以理论上把 `m =` 换成 `line` 或许也能达到效果，但如果不能直接通过 `zcolor` 使得不同高度的颜色不一样（我本以为可以），那干脆直接指定颜色。 
