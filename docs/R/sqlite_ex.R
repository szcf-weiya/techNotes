set.seed(123)
n = 5000000
p = 5
x = matrix(rnorm(n*p), n, p)
x = cbind(1, x)
beta = c(2, rep(1, p))
y = c(x %*% beta) + rnorm(n)

lm(y~0+x)

## solve directly
beta.hat = solve(t(x) %*% x, t(x) %*%y)

## use sqlite
gc()
dat = as.data.frame(x)
rm(x)
gc()
dat$y = y
rm(y)
gc()
colnames(dat) = c(paste0("x", 0:p),"y")
gc()

library(RSQLite)
m = dbDriver("SQLite")
dbfile = "regression.db"
con = dbConnect(m, dbname = dbfile)
if (dbExistsTable(con, "regdata"))
  dbRemoveTable(con, "regdata")
dbWriteTable(con, "regdata", dat, row.names = FALSE)
dbDisconnect(con)
rm(dat)
gc()

## solve beta by sqlite
con = dbConnect(m, dbname = dbfile)
## get variable names
vars = dbListFields(con, "regdata")
xnames = vars[-length(vars)]
yname = vars[length(vars)]
## generate sql statements to compute X'X
mul = outer(xnames, xnames, paste, sep = "*")
lower.idx = lower.tri(mul, diag = TRUE)
mul.lower = mul[lower.idx]
sql = paste0("sum(", mul.lower, ")", collapse = ",")
sql = sprintf("select %s from regdata", sql)
txx.lower = unlist(dbGetQuery(con, sql), use.names = FALSE)
## X'X
txx = matrix(0, p+1, p+1)
txx[lower.idx] = txx.lower
txx = t(txx)
txx[lower.idx] = txx.lower
## Generate SQL statements to compute X'Y
sql = paste(xnames, yname, sep = "*")
sql = paste0("sum(", sql, ")", collapse = ",")
sql = sprintf("select %s from regdata", sql);
txy = unlist(dbGetQuery(con, sql), use.names = FALSE)
txy = matrix(txy, p+1)
## compute beta hat in R
beta.hat.db = solve(txx, txy)

## difference
max(abs(beta.hat-beta.hat.db))
dbDisconnect(con)