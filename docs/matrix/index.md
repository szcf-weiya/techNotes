# Matrix

## Eigenvectors in ellipsoids

1. [Correspondence between eigenvalues and eigenvectors in ellipsoids](https://math.stackexchange.com/questions/581702/correspondence-between-eigenvalues-and-eigenvectors-in-ellipsoids)
2. [Relationship between ellipsoid radii and eigenvalues](https://math.stackexchange.com/questions/80226/relationship-between-ellipsoid-radii-and-eigenvalues/80237#80237)

## symmetric positive semidefinite

1. [Proof of a matrix is positive semidefinite iff it can be written in the form X′X](https://math.stackexchange.com/questions/482688/proof-of-a-matrix-is-positive-semidefinite-iff-it-can-be-written-in-the-form-x)

## Can matrix exponentials be negative?

Refer to [Can matrix exponentials ever be negative? If so, under what conditions?](https://math.stackexchange.com/questions/926943/can-matrix-exponentials-ever-be-negative-if-so-under-what-conditions).

One argument: if the matrix to have non-negative entries off-diagonal, then the matrix exponentials must be positive.

## Spectral Theorem

Refer to [Brillinger, D. R. (1981). Time series: data analysis and theory (Vol. 36). Siam.](https://books.google.com.hk/books?hl=zh-CN&lr=&id=3DFJfgEW94gC&oi=fnd&pg=PR3&dq=+Time+series:+data+analysis+and+theory&ots=WbD7bna2Gk&sig=iJgee3csDKeRp-cJr3eR0OZPiuo&redir_esc=y#v=onepage&q=Time%20series%3A%20data%20analysis%20and%20theory&f=false).

### Theorem 3.7.1

If $\H$ is a $J\times J$ Hermitian matrix, then 

$$
\H = \sum_{j=1}^J\mu_j\U_j\bar\U_j^T
$$

where $\mu_j$ is the $j$-th latent value of $\H$ and $\U_j$ is the corrsponding latent vector.

### Corollary 3.7.1

If $\H$ is $J\times J$ Hermitian, then it may written $\U\M\bar \U^T$ where $\M = \diag\{\mu_j;j=1,\ldots,J\}$ and $\U=[\U_1,\ldots,\U_J]$ is unitary. Also if $\H$ is non-negative definite, then $\mu_j\ge 0,j=1,\ldots,J$.

### Theorem 3.7.2

If $\Z$ is $J\times K$, then 

$$
\Z = \sum_{j\le J,K}\mu_j\U_j\bar\V_j^T
$$

where $\mu_j^2$ is the $j$-th latent value of $\Z\bar\Z^T$ (or $\bar \Z^T\Z$), $\U_j$ is the $j$-th latent vector of $\Z\bar\Z^T$ and $\V_j$ is the $j$-th latent vector of $\bar \Z^T\bar \Z$ and it is understood $\mu_j\ge 0$.

### Corollary 3.7.2

If $\Z$ is $J\times K$, then it may be written $\U\M\bar\V^T$ where the $J\times K$ $\M=\diag\{\mu_j:j=1,\ldots,J\}$, the $J\times J$ $\U$ is unitary and $K\times K$ $\V$ is also unitary.

### Theorem 3.7.4

Let $\Z$ be $J\times K$. Among $J\times K$ matrices $\A$ of rank $L\le J,K$

$$
\mu_j([\Z-\A][\overline{\Z-\A}]^T)
$$

is minimized by 

$$
\A = \sum_{j=1}^L\mu_j\U_j\bar \V_j^T\,.
$$

The minimum achieved is $\mu_{j+L}^2$.

### Corollary 3.7.4

The above choice of $\A$ also minimizes 

$$
\Vert \Z-\A\Vert^2 = \sum_{j=1}^J\sum_{k=1}^K\vert Z_{jk}-A_{jk}\vert^2
$$

for $\A$ of rank $L\le J,K$. The minimum achieved is 

$$
\sum_{j>L}\mu_j^2\,.
$$

## Orthogonal matrix

Orthogonal matrix implies that both of columns and rows are orthogonal.

Refer to [Column Vectors orthogonal implies Row Vectors also orthogonal?](https://math.stackexchange.com/questions/52717/column-vectors-orthogonal-implies-row-vectors-also-orthogonal)