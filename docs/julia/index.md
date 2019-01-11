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


## 关于 `mean()`

1. `using Statistics` 后才能用 `mean()`，而 `using Distributions` 后也能用 `mean()`。前者表示 `generic function with 5 methods`，后者称 `generic function with 78 methods`.

## `Normal` 中标准差为 0 的问题

![](normal-zero-var.png)

可知，最低可以支持 `1e-323`，所以似乎也支持 `sqrt(1e-646)`，但并没有，而且当 `sqrt(1e-324)` 时精度就不够了，似乎 `sqrt(x)` 的精度与 `x` 的精度相当。

## 连等号赋值

如果采用 `a=b=c=ones(10)` 形式赋值的话，则如果后面改变 `a` 的值，`b` 和 `c` 的值也将随之改变。

但如果 `a=b=c=1` 为常值的话，则三个变量的值还是独立的。

## `@distributed`

如果配合 `sharedarrays` 使用时，需要加上 `@sync`, 参考[@fetch](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@fetch)

## ERROR: `expected Type{T}`

参考 [ERROR: LoadError: TypeError: Type{...} expression: expected Type{T}, got Module](https://discourse.julialang.org/t/error-loaderror-typeerror-type-expression-expected-type-t-got-module/1230/4)

其中举了一个小例子

```julia
module Foo end
Foo{Int64}
```

会爆出这样的错误。但是一开始竟然没有仔细类比，最后在 REPL 中逐行试验才发现是，`using SharedArrays` 后直接用 `SharedArrays{Float64}(10)`，这与上面 `Foo` 的错误形式完全一样，竟然没有仔细类比。哎，看来以后多思考一下错误可能的原因，不要一味蛮力试验。

## type, instance, and object

看两个句子：

1. A type union is a special abstract type which includes as objects all instances of any of its argument types
2. `Nothing` is the singleton type whose only instance is the object `nothing`.

从中分析知道，instance 相对于 types，而 object 相对 instance。一个 type 可能有多个 instance，每个 instance 称之为 object。

1. instance of some types
2. object of some instances

## Couldn't find libpython error

```julia
ENV["PYTHON"]=""; Pkg.build("PyCall")
```

to install its own private Miniconda distribution for you, in a way that won't interfere with your other Python installations.

Refer to [Couldn't find libpython error #199](https://github.com/JuliaPy/PyCall.jl/issues/199)

## parallel

references

1. [Julia parallel computing over multiple nodes in cluster](https://stackoverflow.com/questions/43079309/julia-parallel-computing-over-multiple-nodes-in-cluster)
2. [Using julia -L startupfile.jl, rather than machinefiles for starting workers.](https://white.ucc.asn.au/2017/08/17/starting-workers.html)
3. [Help setting up Julia on a cluster](https://discourse.julialang.org/t/help-setting-up-julia-on-a-cluster/5519)

## pbsdsh (unsolved)

Submit a pbsdsh job and specify multiply nodes with multiply cores, say `nodes=2:ppn=4`

error log file (see full file in the cluster: `dshmcmc.e21907`): 

```
fatal: error thrown and no exception handler available.
InitError(mod=:Base, error=ArgumentError(msg="Package Sockets not found in current path:
- Run `Pkg.add("Sockets")` to install the Sockets package.
"))
rec_backtrace at /buildworker/worker/package_linux64/build/src/stackwalk.c:94
record_backtrace at /buildworker/worker/package_linux64/build/src/task.c:246
jl_throw at /buildworker/worker/package_linux64/build/src/task.c:577
require at ./loading.jl:817
init_stdio at ./stream.jl:237
jfptr_init_stdio_4446.clone_1 at /opt/share/julia-1.0.0/lib/julia/sys.so (unknown line)
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2182
reinit_stdio at ./libuv.jl:121
__init__ at ./sysimg.jl:470
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2182
jl_apply at /buildworker/worker/package_linux64/build/src/julia.h:1536 [inlined]
jl_module_run_initializer at /buildworker/worker/package_linux64/build/src/toplevel.c:90
_julia_init at /buildworker/worker/package_linux64/build/src/init.c:811
julia_init__threading at /buildworker/worker/package_linux64/build/src/task.c:302
main at /buildworker/worker/package_linux64/build/ui/repl.c:227
__libc_start_main at /lib64/libc.so.6 (unknown line)
_start at /opt/share/julia-1.0.0/bin/julia (unknown line)
```

But when I just use single node, and arbitrary cores, say `nodes=1:ppn=2`, it works well.

### references

1. [PBSDSH - High Performance Computing at NYU - NYU Wikis](https://wikis.nyu.edu/display/NYUHPC/PBSDSH)
2. [pbsdsh usage](http://docs.adaptivecomputing.com/torque/4-1-3/Content/topics/commands/pbsdsh.htm)

## julia local package 失败折腾记录

1. no error after `add ~/GitHub/adm.jl`, but `using adm` cannot work. refer to [Adding a local package](https://docs.julialang.org/en/latest/stdlib/Pkg/#Adding-a-local-package-1)
2. set `startup.jl` but still not work. refer to [How does Julia find a module?](https://en.wikibooks.org/wiki/Introducing_Julia/Modules_and_packages)
3. one possible way: [Finalizing Your Julia Package: Documentation, Testing, Coverage, and Publishing](http://www.stochasticlifestyle.com/finalizing-julia-package-documentation-testing-coverage-publishing/)

## julia `for` scope

The following code
```julia
i = 0
for j = 1:10
    i = i + 1
end
```
will report 
```julia
ERROR: UndefVarError: i not defined
```

A (possible) reasonable explanation is, $i$ is a global variable, we cannot modify a global variable in a local block without `global` keyword, but we can read `i` in the `for` loop.

Alternatively, we can use `let` block,

```julia
let
i = 0
for j = 1:10
    i = i + 1
end
i
end
```

then `i` isn't really a global variable anymore.

References:

1. [REPL and for loops (scope behavior change)](https://discourse.julialang.org/t/repl-and-for-loops-scope-behavior-change/13514/3)
2. [Scope of variables in Julia](https://stackoverflow.com/questions/51930537/scope-of-variables-in-julia/)
3. [Manual: Scope of Variables](https://docs.julialang.org/en/v1/manual/variables-and-scoping/index.html)

## convert a matrix into an array of array


```julia
mapslices(x->[x], randn(5,5), dims=2)[:]
```

refer to 

[Converting a matrix into an array of arrays](https://discourse.julialang.org/t/converting-a-matrix-into-an-array-of-arrays/17038)

## index from 0

using `OffsetArrays` package, refer to

[Github: OffsetArrays](https://github.com/JuliaArrays/OffsetArrays.jl)