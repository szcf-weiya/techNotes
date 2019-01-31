# Statistics

## Distribution relationships

### Gamma & Wishart

$$
Gamma(k,\theta) = Wishart(\nu, \sigma^2)\,,
$$

where $k=\nu/2$ and $\theta=2\sigma^2$.

Refer to [ARPM: Wishart and Gamma distribution](https://www.arpm.co/lab/GammaDistribNotation.html)

### Gamma & Chi-squared

- $\chi^2_k = Gamma(k/2, 2)$
- $c\chi^2_k = Gamma(k/2, 2c)$
- $\sum_{i=1}^n\chi^2_k(\lambda) = \chi^2_{nk}(n\lambda)$.

## Calculus

1. $\int_0^\infty e^{-x^2}dx$, see [Impossible Integral](https://www.math.hmc.edu/funfacts/ffiles/20008.3.shtml).
2. [Integral Inequality Absolute Value: $|\int^b_af(x)g(x) dx∣\le\int_a^b|f(x)|⋅|g(x)| dx$](https://math.stackexchange.com/questions/429220/integral-inequality-absolute-value-left-int-ab-fx-gx-dx-right)
3. $\sum \dfrac{1}{n\log n}$ and $\sum \dfrac{1}{n\log(\log(\log n))}$ diverges, while $\sum\dfrac{1}{x\log^2x}$ converges, refer to [Infinite series ...](https://math.stackexchange.com/questions/574503/infinite-series-sum-n-2-infty-frac1n-log-n) for more details.

## Simplex Volume

A simplex in $n$-dimensional Euclidean space is a convex solid with $n+1$ vertices, its volume is 

$$
V_n = \frac{1}{n!}h_nh_{n-1}\cdots h_1\,,
$$

where $h_i$ is the distance between the $i$-th and the $i+1$-th vertices.

See [Simplex Volumes and the Cayley-Menger Determinant](https://www.mathpages.com/home/kmath664/kmath664.htm) for more details.

## Interesting Graphs

Cardioid (心形)：$r=2a(1+\cos\theta)$

## Bootstrap samples

1. [How many different bootstrap samples are there?](http://statweb.stanford.edu/~susan/courses/s208/node37.html)

## Confidence set for parameter vector in linear regression

[Confidence set for parameter vector in linear regression](https://stats.stackexchange.com/questions/18322/confidence-set-for-parameter-vector-in-linear-regression)

## Order statistics

1. [Expected value of the k-th order statistic from uniform random variables](https://math.stackexchange.com/questions/2251604/expected-value-of-the-k-th-order-statistic-from-uniform-random-variables)

## Spherical and Elliptical Distributions

[Spherical and Elliptical Distributions](http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode42.html)

## Wishart Distribution

![](wishart.png)