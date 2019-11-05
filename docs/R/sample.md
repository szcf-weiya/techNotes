# `sample` function in R-3.6.0+

Wenyu found a very interesting question about R programming. He cannot reproduce the code in the book, but his sister can successfully produce the results last year. That's so strange. That might be some changing thing, packages or r-base itself.

The program is as follows:

```r
library(tree)
library(ISLR)
attach(Carseats)
High=ifelse(Sales<=8,"No","Yes")
Carseats=data.frame(Carseats,High)

tree.carseats=tree(High~.-Sales,Carseats)
summary(tree.carseats)

Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
```

the book reports

```
##          High.test
## tree.pred No Yes
##       No  86  27
##       Yes 30  57
```

while we get

```
##          High.test
## tree.pred  No Yes
##       No  102  30
##       Yes  15  53
```

## Package versions

All of the related packages are `tree` and `ISLR`, where the second one has not been updated for a long time, and so no need to consider it. For `tree`, there is active updates in these years,

```md
Version 1.0-39  2018-03-17

Allow more C-level space for labels

Version 1.0-38  2018-03-14

Tweak to cv.tree.

Version 1.0-37  2016-01-20

Include <stdio.h>, not relying on R.h

Version 1.0-36  2015-06-29

Add tests/deparse.R .
Update reference output.
Tweak NAMESPACE imports.
```

but actually the last version is back to March 2018, which is already prior to Wenyu's sister. Anyway, we have tried to install the old version package, and try to see some differences,

The command for installing a specified version would be

```
install.packages("https://cran.r-project.org/src/contrib/Archive/tree/tree_1.0-39.tar.gz", repos = NULL, type = "source")
```

in the R console. But at the beginning, I am not familiar with it. So I use the interface of rstudio to manually install the old-version `tree`. Actually, I had tried `v1.0-28`, `v1.0-33`, `v1.0-37` and `v1.0-39`.

## R version

Currently, we are using the latest R version, v3.6.0. Firstly, I had tried the old version `v3.4.4` on my server version rstudio. It can successfully reproduce the results.

Now our goal is to determine the version from which we fail to obtain the same results. It requires us to install multiple r versions to test the program.

The `docker` facilitates the multiple R version easily. Here is an official images for r-base, including all old versions of R. We can install a specified R version with the following commands by tagging the version number, 

```
docker pull r-base:3.6.0
docker run -it --rm r-base:3.6.0
```

As a result, `v3.5.3` can reproduce the results, while it failed in the R version from `v3.6.0`. Then I found the changelog for `v3.6.0`.

The first one is exactly what we want,


> **Changes to random number generation.** R 3.6.0 changes the method used to generate random integers in the sample function. In prior versions, the probability of generating each integer could vary from equal by up to 0.04% (or possibly more if generating more than a million different integers). This change will mainly be relevant to people who do large-scale simulations, and it also means that scripts using the sample function will generate different results in R 3.6.0 than they did in prior versions of R. If you need to keep the results the same (for reproducibility or for automated testing), you can revert to the old behavior by adding `RNGkind(sample.kind="Rounding")`) to the top of your script.

In a word, for the R starting from v3.6.0, we can add

```r
RNGkind(sample.kind="Rounding")
```

to reproduce the results that did in prior versions of R.

