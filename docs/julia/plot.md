# 作图

## 等高线图 (contour)

```julia
using Plots
using Distributions

x_grid = range(-2, 2, length=100)
y_grid = range(-2, 2, length=100)
Sigma = [1 0.9; 0.9 1];
contour(x_grid, y_grid, (x, y)->pdf(MvNormal([0, 0], Sigma), [x, y]), cbar=false)
```