using CmdStan
src = "
data {
    int <lower=0> N;
    int <lower=0> y[N];
}
parameters {
    real<lower=0,upper=1> theta;
    real<lower=0> lambda;
}
model {
    for (n in 1:N){
        if (y[n] == 0)
            target += log_sum_exp(bernoulli_lpmf(1 | theta), bernoulli_lpmf(0 | theta) + poisson_lpmf(y[n] | lambda));
        else
            target += bernoulli_lpmf(0 | theta) + poisson_lpmf(y[n] | lambda);
    }
}
"

model = Stanmodel(model = src)
using Distributions
y = zeros(Int, 100)
y[1:80] = rand(Poisson(3),80)
data = Dict("N" => 100, "y" => y)
res = stan(model, data)