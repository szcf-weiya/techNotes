---
comments: true
hide:
    - toc
---

# Comparisons

- R in Julia: `RCall.jl`
- Python in Julia: `PyCall.jl`
- Python in R: `reticulate`

LANG | Julia | R | Python | Others
-- | -- | -- | -- | -- 
[padding zero on the left](../../julia/#padding-zero-on-the-left) | `lpad(1, 3, '0')` | `sprintf("%03d", 1)` | `f"{1:03}"` | `printf "%03d" 1` (Shell)
[check if substring](../../julia/#check-if-substring) | `occursin()` | | `in` |
[sum an array along row](../../julia.md#dims1) | `dims=1` | `margin=2` | `axis=0` | 
[read image](../../python/opencv/#read-image) | `Images::load()` | | `cv2.imread()` <br/> `skimage.io.imread()` |`imread()` (Matlab)
[write image](../../python/opencv/#read-image) | `save()` | | `cv2.imwrite()` | 
[`sort` and `argsort`](../../R/#sort-rank-order) | `sort, sortperm` | `sort(), order()` | `sorted(), np.argsort()` | 
[Jupyter kernels](../../python/#different-kernels) | `IJulia` | `IRkernel` | `ipykernel` | 
[too many open figures](../../julia/#gr-too-many-open-files) | `inline` | | `rcParams` | 
[merge multiple slices](../../python.md#merge-multiple-slices) | `vcat` | | `np.r_` |
index of true elements | `findall` | `which` | `np.where()[0]` | 
freq table | `StatsBase::countmap()` | `table`| `np.unique(return_counts=True)`| 
contingency table | `FreqTable::freqtable()` | `table` | |
figure size | `size in pixel` | | `figsize in inch` | 
[straight line](https://stackoverflow.com/questions/55427314/whats-julias-plots-jls-equivalent-of-rs-abline) | `Plots.abline!()` | `abline()` | |
get unique elements | `unique(x)` | `unique(x)` |  |
index for unique elements | `unique(i->x[i], 1:length(x))` | `which(!duplicated(x))` | |
layout | `@layout` | [`layout`](../../R/plot/) | |
priority of `:` | `0.1+1:10-1 == (0.1+1):(10-1)`  | `0.1+1:10-1 == 0.1+(1:10)-1` | |
enumerate | `for (k, v) in enumerate(d)` | | `for k, v in enumerate(d):` |
[keep dims](../../R/) | `a[1:1,:]` | `a[1,,drop=FALSE]` | `a[0:1,:]` | 
expand dims | | | `a[:, None]` <br/> `np.expand_dims(a, axis=-1)` <br/> `a[:, np.newaxis]` |
default color | [`palette(:default)`](https://github.com/szcf-weiya/ESL-CN/blob/6ba63dc8cddf0406c4d5e07166b46c81f37e7993/imgs/fig.14.30/kpca.jl#L12) | | [`plt.rcParams['axes.prop_cycle'].by_key()['color']`](https://statisticaloddsandends.wordpress.com/2023/05/24/getting-matplotlibs-default-colors/) | 
