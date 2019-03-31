using CmdStan
src = "
data {
    int<lower=0> N;
    int<lower=0> y[N];
}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0> lambda;
}
model {
    for (n in 1:N) {
        if (y[n] == 0)
            1 ~ bernoulli(theta);
        else {
            0 ~ bernoulli(theta);
            y[n] ~ poisson(lambda) T[1,];
        }
    }
}
"
src1 = "
data {
    int<lower=0> N;
    int<lower=0> y[N];
}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0> lambda;
}
model {
    for (n in 1:N){
        if (y[n] == 0)
            target += log(theta);
        else
            target += log1m(theta) + poisson_lpmf(y[n] | lambda) - poisson_lccdf(0 | lambda);
    }
}
"
src2 = "
data {
    int<lower=0> N;
    int<lower=0> y[N];
}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0> lambda;
}
model {
    for (n in 1:N){
        if (y[n] == 0)
            target += log(theta);
        else
            target += log1m(theta) + poisson_lpmf(y[n] | lambda) - log1m_exp(-lambda);
    }
}
"
src3 = "
functions {
    int num_zero(int[] y) {
        int nz = 0;
        for (n in 1:size(y))
            if (y[n] == 0)
                nz += 1;
        return nz;
    }
}
data {
    int<lower=0> N;
    int<lower=0> y[N];
}
transformed data {
    int<lower=0, upper=N> N0 = num_zero(y);
    int<lower=0, upper=N> Ngt0 = N - N0;
    int<lower=1> y_nz[N - num_zero(y)];
    {
        int pos = 1;
        for (n in 1:N) {
            if (y[n] != 0) {
                y_nz[pos] = y[n];
                pos += 1;
            }
        }
    }
}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0> lambda;
}
model {
    N0 ~ binomial(N, theta);
    y_nz ~ poisson(lambda);
    target += -Ngt0 * log1m_exp(-lambda);
}
"

model = Stanmodel(model = src3)
using Distributions
N = 200
y = zeros(Int, N)
y[1:140] = rand(Poisson(8), 140)
data = Dict("N" => 200, "y" => y)
res = stan(model, data)