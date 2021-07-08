---
hide:
    - toc
---

# Comparisons

LANG | Julia | R | Python | Matlab
-- | -- | -- | -- | -- 
[padding zero on the left](../../julia/#padding-zero-on-the-left) `1 -> 001` | `lpad(1, 3, '0')` |  | `f"{1:03}"` | 
[check if substring](../../julia/#check-if-substring) | `occursin()` | | `in` |
[sum an array along row](../../julia/#dims1) | `dims=1` | `margin=2` | `axis=0` | 
[read image](../../python/opencv/#read-image) | `Images::load()` | | `cv2.imread()`/`skimage.io.imread()` |`imread()`
[`sort` and `argsort`](../../R/#sort-rank-order) | | `sort(), order()` | `sorted(), np.argsort()` | 
[Jupyter kernels](../../python/#different-kernels) | `IJulia` | `IRkernel` | `ipykernel` | 
[too many open figures](../../julia/#gr-too-many-open-files) | `inline` | | `rcParams` | 

