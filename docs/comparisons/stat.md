---
hide:
    - toc
---

# Comparisons

LANG | Julia | R | Python | Others
-- | -- | -- | -- | -- 
[padding zero on the left](../../julia/#padding-zero-on-the-left) | `lpad(1, 3, '0')` |  | `f"{1:03}"` | `printf "%03d" 1` (Shell)
[check if substring](../../julia/#check-if-substring) | `occursin()` | | `in` |
[sum an array along row](../../julia/#dims1) | `dims=1` | `margin=2` | `axis=0` | 
[read image](../../python/opencv/#read-image) | `Images::load()` | | `cv2.imread()`/`skimage.io.imread()` |`imread()` (Matlab)
[write image](../../python/opencv/#read-image) | `save()` | | `cv2.imwrite()` | 
[`sort` and `argsort`](../../R/#sort-rank-order) | `sort, sortperm` | `sort(), order()` | `sorted(), np.argsort()` | 
[Jupyter kernels](../../python/#different-kernels) | `IJulia` | `IRkernel` | `ipykernel` | 
[too many open figures](../../julia/#gr-too-many-open-files) | `inline` | | `rcParams` | 
[merge multiple slices](../../python/#merge-multiple-slices) | `vcat` | | `np.r_` |
index of true elements | `findall` | `which` | `np.where()[0]` | 
freq table | | `table`| `np.unique(return_counts=True)`| 