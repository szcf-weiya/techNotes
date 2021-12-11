using Random

function f(seed)
    println(rand(MersenneTwister(seed)))
end

function g(seed)
    println(rand())
    f(seed)
    println(rand())
end