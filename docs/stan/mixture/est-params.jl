using CmdStan
src = "
data {
    int<lower=1> K;
    int<lower=1> N;
    real y[N];
}
parameters {
    simplex[K] theta;
    ordered[K] mu;
    vector<lower=0>[K] sigma;
}
model {
    vector[K] log_theta = log(theta);
    sigma ~ lognormal(0, 2);
    mu ~ normal(0, 10);
    for (n in 1:N){
        vector[K] lps = log_theta;
        for (k in 1:K)
            lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
        target += log_sum_exp(lps);
    }
}
"

using Distributions
N = 100
K = 3
y1 = rand(Normal(-10, 7), 30)
y2 = rand(Normal(20, 8), 40)
y3 = rand(Normal(40, 9), 30)
y = vcat(y1, y2, y3)
data = Dict("N" => N, "K" => K, "y" => y)
model = Stanmodel(model = src)
res = stan(model, data)


