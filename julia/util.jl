using Primes

function save_plots(ps)
    n = length(ps)
    for (i, p) in enumerate(ps)
        savefig(p, "/tmp/p$i.pdf")
    end
    fignames = "/tmp/p" .* string.(1:n) .* ".pdf"
    run(`pdftk $fignames cat output /tmp/all.pdf`)
end

function save_grid_plots(ps, out = "all")
    n = length(ps)
    res = factor(Vector, n)
    # determine nrow and ncol of the grid
    l = length(res)
    nrow = res[1]
    ncol = 1
    if l == 2
        nrow = res[2]
        ncol = res[1]
    else
        ncol = res[end]
        nrow = Int(n / ncol)
    end
    save_grid_plots(ps, nrow, ncol, out)
end

function save_grid_plots(ps, nrow, ncol, out = "all")
    n = length(ps)
    if nrow * ncol != n
        error("different number of plots")
    end
    for (i, p) in enumerate(ps)
        savefig(p, "/tmp/p$i.png")
    end
    for i = 1:nrow
        fignames = "/tmp/p" .* string.(ncol * (i-1) .+ (1:ncol)) .* ".png"
        run(`convert $fignames +append /tmp/pp$i.png`)
    end
    for i = 1:nrow
        fignames = "/tmp/pp" .* string.(1:nrow) .* ".png"
        run(`convert $fignames -append /tmp/$out.png`)
    end
end

function myparse(str)
    if occursin(":", str) # only for int currently
        ys = split(str, ":")
        xs = parse.(Int, ys)
        if length(xs) == 2
            return xs[1]:xs[2]
        else
            return xs[1]:xs[2]:xs[3]
        end
    else
        if occursin(".", str)
            return parse(Float64, str)
        else
            return parse(Int, str)
        end
    end
end

"""
    split_keystr(str)

Split string with format `StrNum_StrNum` into key pairs

```jldoctest
julia> split_keystr("a123_b234")
Dict{Any, Any} with 2 entries:
  "b" => 234
  "a" => 123
```
"""
function split_keystr(x)
    # x = "a123_b234"
    xs = split(x, "_")
    res = Dict()
    for y in xs
        try
            ys = match(r"([a-zA-Z]+)(.*)", y).captures
            if ys[2] != ""
                res[ys[1]] = myparse(ys[2])
            end
        catch e
            @warn e
        end
    end
    return res
end
