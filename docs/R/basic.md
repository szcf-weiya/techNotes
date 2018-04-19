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

