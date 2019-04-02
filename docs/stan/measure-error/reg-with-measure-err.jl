using CmdStan
src = "
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta*x, sigma);
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma ~ cauchy(0, 5);
}
"

src1 = "
data {
    ...
    real x_meas[N]; // measurement of x 
    real<lower=0> tau; // measurement noise
}
parameters {
    real x[N]; // unknown true value
    real mu_x; // prior location
    real sigma_x; // prior scale
    ...
}
model {
    x ~ normal(mu_x, sigma_x); //prior
    x_meas ~ normal(x, tau); // measurement model
    y ~ normal(alpha + beta * x, sigma);
    ...
}
"