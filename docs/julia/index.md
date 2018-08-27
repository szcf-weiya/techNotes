# julia相关

[主页](https://julialang.org/)

## Is there a way to undo `using` in Julia?

NO

1. [https://stackoverflow.com/questions/33927523/can-i-make-julia-forget-a-method-from-the-repl](https://stackoverflow.com/questions/33927523/can-i-make-julia-forget-a-method-from-the-repl)
2. [https://stackoverflow.com/questions/36249313/is-there-a-way-to-undo-using-in-julia](https://stackoverflow.com/questions/36249313/is-there-a-way-to-undo-using-in-julia) 


## ->

[Anonymous Functions](https://docs.julialang.org/en/v0.6.1/manual/functions/#man-anonymous-functions-1)

## plot

1. [Tutorial for plots](http://docs.juliaplots.org/latest/tutorial/)


## 几个发现

1. `using Statistics` 后才能用 `mean()`，而 `using Distributions` 后也能用 `mean()`。前者表示 `generic function with 5 methods`，后者称 `generic function with 78 methods`.
