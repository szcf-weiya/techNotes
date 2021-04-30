# 一道与 factor 有关的问题 

W 问了个问题，

> 问题：对于一个4维向量A=(a1,a2,a3,a4)，我想要将其转化为另一个4维向量B=(b1,b2,b3,b4)，b1=1，然后对于b_i，如果a_i在A的前i-1个元素里没出现过，那么b_i=max(b[1:(i-1)])+1，否则b_i=出现过的那个元素a_k对应的b_k。
例子：(23,12,23,13) → (1,2,1,3), (20,20,20,20)→(1,1,1,1)

因为逻辑挺清晰的，所以第一感觉便是自己写函数，于是随手写了个 Julia 版本的，

```julia
--8<-- "docs/julia/factor.jl"
```

其中 `Dict` 主要是为了缩小搜索空间。不过 W 指出在 R 中自己写函数挺费时的，然后他自己想出了个很优雅的解法

```R
ff <- function(A) { as.numeric(factor(A,levels = unique(A))) }
```

因为自己想复习下 Rcpp，所以将上述 Julia 代码改写成 Rcpp，

```cpp
--8<-- "docs/R/factor.cpp"
```

然后与 W 的方法进行比较，部分测试结果如下，

![](https://user-images.githubusercontent.com/13688320/116717848-ed728680-aa0b-11eb-9f6a-271296068e48.png)

可见 Rcpp 版本 `f` 略优于 W 的方法 `ff`。另外，也将其与 Julia 版本进行比较，

![image](https://user-images.githubusercontent.com/13688320/116718480-95884f80-aa0c-11eb-8952-1434126784c4.png)

有点惊讶，速度竟然快这么多！
