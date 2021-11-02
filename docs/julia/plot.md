# Plot

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

## axis off

```julia
plot(..., axis = nothing)
```

and similar grammar is

```julia
plot(..., ticks=nothing, border=nothing)
```

refer to [How can I implement "axis off"? #428](https://github.com/JuliaPlots/Plots.jl/issues/428)

[In practice](https://github.com/szcf-weiya/Cell-Video/blob/577a1558e93970414ea10fc0e5012905d50cf765/realdata/run.jl#L278), I adopted `axis = nothing, border = :none`.

## plot kernel density

refer to [Kernel density estimation status](https://discourse.julialang.org/t/kernel-density-estimation-status/5928)

## PyPlot

- [grid_plot_acc_vs_rate_revisit](https://github.com/szcf-weiya/Cell-Video/blob/8cffd45451c0b1af9da4199c7ef611d836c0e86e/DP/visualization.jl#L155-L217) and [demo](https://github.com/szcf-weiya/Cell-Video/blob/last-revisit/DP/110_original_with_revisit_2019-10-09T11:20:28_oracle_setting_2019-09-22T20:06:59_precision.pdf)
    - multiple subplots with `sharex="all", sharey="all"`
    -  `ax0 = fig.add_subplot(111, frameon=false)`
    - `plt.text`

## multiple labels

In Julia 1.4.0 with Plots.jl v1.0.14,

```julia
using Plots
x = rand(10, 2)
plot(1:10, x, label = ["a", "b"])
```

will produce

![](labels_col.png)

where these two lines share the same label instead of one label for one line. But if replacing the column vector with row vector,

```julia
plot(1:10, x, label = ["a" "b"])
```

will return the correct result,

![](labels_row.png)

Refer to [Plots (plotly) multiple series or line labels in legend](https://discourse.julialang.org/t/plots-plotly-multiple-series-or-line-labels-in-legend/13001), which also works `GR` backend.

## suptitle for subplots

currently， no a option to set a suptitle for subplots, but we can use `@layout` to plot the title in a grid, such as [szcf-weiya/TB](https://github.com/szcf-weiya/TB/blob/c332307263cdbab20a453e6abe74790236321048/CFPC/sim_cpc_scores.jl#L87-L93)

refer to [Super title in Plots and subplots](https://discourse.julialang.org/t/super-title-in-plots-and-subplots/29865/4)

## PGFPlotsX

Tips:

- [:octicons-issue-closed-16:](https://github.com/szcf-weiya/Clouds/issues/35) do not set `pgfplotsx()` in a function , otherwise it throws

```julia
ERROR: MethodError: no method matching _show(::IOStream, ::MIME{Symbol("application/pdf")}, ::Plots.Plot{Plots.PGFPlotsXBackend}) 
```

- [:octicons-issue-closed-16:](https://github.com/szcf-weiya/Clouds/issues/53) rebuild `GR` if `ERROR: could not load library "libGR.so"`. Possible reason is that the version (such as `BwGt2`) to use has not been correctly built, although it was working well in other versions, such as `yMV3y`.

- []()

when using layout with setting like `@layout([a{0.05w} b; _ c{0.05h}])`

```julia
! Package pgfplots Error: Error: Plot width `15.99043pt' is too small. This can
not be implemented while maintaining constant size for labels. Sorry, label sizes 
are only approximate. You will need to adjust your width..
```

set a larger size, say, increasing 0.05 to 0.2.

## GR: Too many open files

The complete error message is 

```julia
No XVisualInfo for format QSurfaceFormat(version 2.0, options QFlags<QSurfaceFormat::FormatOption>(), depthBufferSize -1, redBufferSize 1, greenBufferSize 1, blueBufferSize 1, alphaBufferSize -1, stencilBufferSize -1, samples -1, swapBehavior QSurfaceFormat::SwapBehavior(SingleBuffer), swapInterval 1, profile  QSurfaceFormat::OpenGLContextProfile(NoProfile))
Falling back to using screens root_visual.
socket: Too many open files
GKS: can't connect to GKS socket application
```

see [the private repo](https://github.com/szcf-weiya/Clouds/issues/34) for more details. The issue has been discussed in [GKS file open error: Too many open files #1723](https://github.com/JuliaPlots/Plots.jl/issues/1723), the solution is

```julia
GR.inline("png")
```

I also came across similar issue with `matplotlib.pyplot`, but it just throws a warning, and one solution is to set

```python
plt.rcParams.update({'figure.max_open_warning': 0})
```

refer to [warning about too many open figures](https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures)

## legend

- `legendtitle`
- as a subplot: `plot(p1, p2, p3, p4, plegend, layout = @layout([a b [c{0.6h}; [d e{0.3w}]] ]))` ([complete example](https://github.com/szcf-weiya/Cell-Video/blob/4721ef10b6f77f59dbed639c6806faa1b644ba06/DP/visualization.jl#L610))