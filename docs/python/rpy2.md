# rpy2

初次体验 `rpy2`

## 安装

尝试过两种安装方法

### 1

`conda install rpy2`: 除了装 `rpy2`，也会装 `r-base` 及一些 R 中的核心包，即便系统中已经有 `R` 了。具体如下

```
Downloading and Extracting Packages
pango-1.42.4         | 528 KB    | ##################################### | 100% 
curl-7.65.3          | 141 KB    | ##################################### | 100% 
r-bit-1.1_14         | 243 KB    | ##################################### | 100% 
r-plogr-0.2.0        | 20 KB     | ##################################### | 100% 
certifi-2019.9.11    | 154 KB    | ##################################### | 100% 
gfortran_linux-64-7. | 10 KB     | ##################################### | 100% 
r-dbi-1.0.0          | 916 KB    | ##################################### | 100% 
r-r6-2.4.0           | 68 KB     | ##################################### | 100% 
ca-certificates-2019 | 131 KB    | ##################################### | 100% 
fribidi-1.0.5        | 112 KB    | ##################################### | 100% 
libssh2-1.8.2        | 250 KB    | ##################################### | 100% 
r-assertthat-0.2.1   | 74 KB     | ##################################### | 100% 
r-bh-1.69.0_1        | 10.9 MB   | ##################################### | 100% 
r-rlang-0.3.4        | 1.0 MB    | ##################################### | 100% 
r-crayon-1.3.4       | 757 KB    | ##################################### | 100% 
make-4.2.1           | 429 KB    | ##################################### | 100% 
r-pillar-1.3.1       | 180 KB    | ##################################### | 100% 
krb5-1.16.1          | 1.4 MB    | ##################################### | 100% 
r-dbplyr-1.4.0       | 622 KB    | ##################################### | 100% 
libcurl-7.65.3       | 588 KB    | ##################################### | 100% 
r-rsqlite-2.1.1      | 1010 KB   | ##################################### | 100% 
r-cli-1.1.0          | 189 KB    | ##################################### | 100% 
tktable-2.10         | 88 KB     | ##################################### | 100% 
_r-mutex-1.0.0       | 2 KB      | ##################################### | 100% 
r-tidyselect-0.2.5   | 138 KB    | ##################################### | 100% 
r-utf8-1.1.4         | 159 KB    | ##################################### | 100% 
r-pkgconfig-2.0.2    | 25 KB     | ##################################### | 100% 
r-prettyunits-1.0.2  | 38 KB     | ##################################### | 100% 
r-bit64-0.9_7        | 487 KB    | ##################################### | 100% 
bwidget-1.9.11       | 113 KB    | ##################################### | 100% 
r-rcpp-1.0.1         | 3.3 MB    | ##################################### | 100% 
r-fansi-0.4.0        | 193 KB    | ##################################### | 100% 
r-blob-1.1.1         | 40 KB     | ##################################### | 100% 
r-digest-0.6.18      | 155 KB    | ##################################### | 100% 
r-memoise-1.1.0      | 45 KB     | ##################################### | 100% 
r-dplyr-0.8.0.1      | 1.9 MB    | ##################################### | 100% 
gfortran_impl_linux- | 9.0 MB    | ##################################### | 100% 
r-purrr-0.3.2        | 397 KB    | ##################################### | 100% 
r-glue-1.3.1         | 165 KB    | ##################################### | 100% 
r-magrittr-1.5       | 173 KB    | ##################################### | 100% 
rpy2-2.9.4           | 272 KB    | ##################################### | 100% 
r-base-3.6.0         | 39.4 MB   | ##################################### | 100% 
r-tibble-2.1.1       | 316 KB    | ##################################### | 100%
```

### 2

`pip install --user rpy2`: 则只会装 rpy2。

```
Successfully installed MarkupSafe-1.1.1 atomicwrites-1.3.0 attrs-19.3.0 cffi-1.13.2 importlib-metadata-0.23 jinja2-2.10.3 more-itertools-7.2.0 packaging-19.2 pluggy-0.13.0 py-1.8.0 pycparser-2.19 pytest-5.2.2 rpy2-3.2.2 simplegeneric-0.8.1 tzlocal-2.0.0 wcwidth-0.1.7 zipp-0.6.0
```

但是使用时，没找到 `libR.so`，猜测可能是因为没有指定 `--enable-R-shlib` [^1][^2]

[^1]: https://stackoverflow.com/questions/51622357/compiling-r-3-5-1-from-source-no-libr-so/51626460#51626460
[^2]: https://support.rstudio.com/hc/en-us/articles/218004217-Building-R-from-source


results list name

res = 

`res.names` can show the array of names, but how to get the element by the name, i.e., like `res$somename` in R.