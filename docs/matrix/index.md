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

## Decomposition

### Cholesky 

A symmetric, positive definite square matrix $A$ has a Cholesky decomposition into a product of a lower triangular matrix $L$ and its transpose $L^T$,

$$
A=LL^T\,.
$$

This decomposition can be used to convert the linear system $Ax=b$ into a pair of triangular systems, $Ly=b,L^Tx=y$, which can be solved by forward and back-substitution.

If the matrix $A$ is near singular, it is sometimes possible to reduce the condition number and recover a more accurate a more accurate solution vector $x$ by scaling as

$$
(SAS)(S^{-1}x) = Sb
$$

where $S$ is a diagonal matrix whose elements are given by $S_{ii}=1/\sqrt{A_{ii}}$. This scaling is also known as **Jacobi preconditioning**.

## QR 

A general rectangular $M$-by-$N$ matrix $A$ has a QR decomposition into the product of an orthogonal $M$-by-$M$ square matrix $Q$, where $Q^TQ=I$, and an $M$-by-$N$ right-triangular matrix $R$,

$$
A=QR\,.
$$

This decomposition can be used to convert the linear system $Ax=b$ into the triangular system $Rx=Q^Tb$, which can be solved by back-substitution.

Another use of the QR decomposition is to compute an orthonormal basis for a set of vectors. The first $N$ columns of $Q$ form an orthonormal basis for the range of $A$, $ran(A)$, when $A$ has full column rank.

## $\rk(AB) \leq \rk(A)$

The range of the matrix $M$ is 

$$
\calR(M)=\{\mathbf{y} \in \R^m \mid \mathbf{y}=M\mathbf{x} \text{ for some } \mathbf{x} \in \R^n\}.
$$

Recall that the rank of a matrix $M$ is the dimension of the range $R(M)$ of the matrix M.
So we have

$$
\rk(AB)=\dim(\calR(AB)), \quad \rk(A)=\dim(\calR(A)).
$$

In general, if a vector space $V$ is a subset of a vector space $W$, then we have

$$
\dim(V) \leq \dim(W).
$$

Thus, it suffices to show that the vector space $\calR(AB)$ is a subset of the vector space $\calR(A)$.

Consider any vector $\y\in\calR(AB)$. Then there exists a vector $\x\in\R^l$ such that $\y=(AB)\x$ by the definition of the range.

Let $\z=B\x\in\R^n$.

Then we have

$$
\mathbf{y}=A(B\mathbf{x})=A\mathbf{z}
$$

and thus the vector $\y$ is in $\calR(A)$. Thus $\calR(AB)$ is a subset of $\calR(A)$ and we have

$$
\rk(AB)=\dim(\calR(AB)) \leq \dim(\calR(A))=\rk(A).
$$

refer to [Rank of the Product of Matrices AB is Less than or Equal to the Rank of A](https://yutsumura.com/rank-of-the-product-of-matrices-ab-is-less-than-or-equal-to-the-rank-of-a/)

## $\rk(A)=\rk(AA^T)=\rh(A^TA)$

Note that the left null space (the null space of $A^T$) is the orthogonal complement to the column space of $A$, that is,

$$
\mathrm{Nul}(A^T)^{\perp} = \mathrm{Col}(A), \quad \mathrm{Nul}(A)^{\perp} = \mathrm{Col}(A^T).
$$

Therefore, $\mathrm{Nul}(A^T)\cap \mathrm{Col}(A)$, and so forth.

Now consider the matrix $A^TA$. Then 

$$
\mathrm{Col}(A^TA) = \{A^TAx\} = \{A^Ty:y\in \mathrm{Col}(A)\}.
$$

But since the null space of $A^T$ only intersects trivially with $\mathrm{Col}(A)$, then $\mathrm{Col}(A^TA)$ must have the same dimension as $\mathrm{Col}(A)$, which gives us the equality of ranks.

Refer to [Christopher A. Wong (https://math.stackexchange.com/users/22059/christopher-a-wong), Rank of product of a matrix and its transpose, URL (version: 2012-10-16)](https://math.stackexchange.com/q/215147), maybe also [Prove $\rk(A^TA)=\rk(A)$ for any $A\in M_{m\times n}$](https://math.stackexchange.com/questions/349738/prove-operatornamerankata-operatornameranka-for-any-a-in-m-m-times-n)

## Only real eigenvalues in symmetric matrices

refer to [The Case of Complex Eigenvalues](http://www.sosmath.com/matrix/eigen3/eigen3.html)

## Calculate the Inverse of a 3x3 matrix

By **Adjugate Matrix**,

1. Check the determinant of the matrix.
2. Transpose the original matrix.
3. Find the determinant of each of the 2x2 minor matrices. 
4. Create the matrix of cofactors.  (assign a sign)
5. Divide each term of the adjugate matrix by the determinant. 

By **Linear Row Reduction**

Refer to [How to Find the Inverse of a 3x3 Matrix](https://www.wikihow.com/Find-the-Inverse-of-a-3x3-Matrix)
