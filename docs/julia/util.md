# A Collection of Miscellaneous Functions

I found that I always rewrite similar functions (if not exactly the same) in different projects. Inspired by [Yihui's xfun R package](https://yihui.org/xfun/), I try to declutter those common functions.

## Save Multiple Plots as PDF

Suppose your figure depends on some parameters, and you want to investigate different parameters. 

```julia
include("util.jl")
function myplot(i)
    plot(title = "$i")
end
save_plots([myplot(i) for i = 1:3])
```

The resulting file is `/tmp/all.pdf`.