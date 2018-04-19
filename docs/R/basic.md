## scale or not

下面两个结果是一样的。

```r
m = matrix(rnorm(4*6, 2), ncol = 4)
cov(m)
sm = scale(m, scale = FALSE)
t(sm) %*% sm/5
```

其中需要注意的地方

- `scale = FALSE`
- `/5`

## PCA & SVD

参考[Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)

