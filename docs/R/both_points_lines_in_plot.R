set.seed(0); x1 <- rnorm(10); x2 <- rnorm(10); x3 <- rnorm(10)
plot(x1, type = "b", pch = 19, lty = 1, col = 1,
     ylim = range(c(x1,x2,x3)))  ## both points and lines
points(x2, pch = 19, col = 2)  ## only points
lines(x3, lty = 2, col = 3)  ## only lines
legend(6, 0.9*max(c(x1,x2,x3)), legend = c("x1", "x2", "x3"),
       pch = c(19, 19, NA), lty = c(1, NA, 2),
       col = c(1,2,3), text.col = c(1,2,3))