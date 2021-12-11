using Random

function f(seed)
    println(rand(Random.seed!(seed)))
end

function g(seed)
    println(rand())
    f(seed)
    println(rand())
end