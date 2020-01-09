plot.new()
plot.window(xlim = c(0, 1), ylim = c(-0.1, 0.9))
abline(h = 0)
abline(v = 0.5)
lines(c(0, 0.5), c(0.3, 0.3), lw = 2, col = "blue")
lines(c(0.5, 0.8), c(0.3, 0), lw = 2, col = "blue")
lines(c(0, 0.8), c(0.3, 0.3), lw = 2)
lines(c(0.8, 0.8), c(0.3, 0), lw = 2)
text(0.8, -0.05, expression(rho), cex = 2)
#axis(side = 1, 0.8, labels = expression(rho), tick = FALSE)
# legend(0.7, 0.8, c(expression(phi[rho]), expression("1( <=" * rho * ")")), col = c("blue", "black"), lw = 2)
legend(0.65, 0.8, c(TeX("$\\phi_\\rho$"), TeX("$1(\\cdot \\leq \\rho)$")), 
       col = c("blue", "black"), lw = 2, cex = 1.2) # here NOT to use `\le`