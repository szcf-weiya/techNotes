# 

step 1:

```
docker pull r-base
```

for specified version,

```
docker pull r-base:3.6.0
```

step 2:

```
docker run -it --rm r-base:3.6.0
```

install.packages("https://cran.r-project.org/src/contrib/Archive/tree/tree_1.0-39.tar.gz", repos = NULL, type = "source")