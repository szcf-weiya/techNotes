using Random
import StatsBase.sample
function g(seed = 1234)
    println(sample(Random.seed!(seed), 1:5, 5))
end
