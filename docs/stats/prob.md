## the product of Gaussian

- with any hypothesis, the product of Gaussian is not Gaussian, take $Y=X$ as an example
- if $X, Y$ are independent, it is still not Gaussian. Consider

$$
Z = \frac{X^2-Y^2}{2} = \frac{X-Y}{\sqrt 2} \frac{X+Y}{\sqrt 2}
$$

then $Z$ is the product of two independent Gaussian, but the characteristic function of $Z$ is

$$
\begin{align*}
\varphi_Z(t) &= E\exp\left(\frac{X^2-Y^2}{2}\right) \\
&= E\exp\left(1-2i\frac t2\right)^{-1/2}\exp\left(1-2i\frac{-t}{2}\right)^{-1/2} \\
&= (1-it)^{-1/2}(1+it)^{-1/2}=(1+t^2)^{-1/2}\,.
\end{align*}
$$

- the distribution can be directly called as [Normal Product Distribution](https://mathworld.wolfram.com/NormalProductDistribution.html)

refer to [Is the product of two Gaussian random variables also a Gaussian?](https://math.stackexchange.com/questions/101062/is-the-product-of-two-gaussian-random-variables-also-a-gaussian)

## functions of independent random variables

it is trivial with sigma algebra.

refer to [Functions of Independent Random Variables](https://stats.stackexchange.com/questions/94872/functions-of-independent-random-variables)