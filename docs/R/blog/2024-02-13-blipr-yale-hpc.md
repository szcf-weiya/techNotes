---
comments: true
---

# Installation of `blipr` on Yale HPC

The login node is not allowed to perform heavy operation, and the process might be killed if you directly perform the installation. A better way is to request a work node,

```bash
XXX@login1.mccleary ~]$ srun -p pi_zhao --pty bash -i
srun: job 21544098 queued and waiting for resources
srun: job 21544098 has been allocated resources
```

then load an R env, and attempt to install the [`blipr` package from GitHub](https://github.com/amspector100/blipr)

```bash
$ module load R
$ R
> remotes::install_github("amspector100/blipr")
```

!!! failure "requirement of R package `rcbc`"
    ```r
    ERROR: dependency ‘rcbc’ is not available for package ‘blipr’
    * removing ‘/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/blipr’
    Warning message:
    In i.p(...) :
    installation of package ‘/tmp/Rtmpb8HWYq/file26b46d3bedf90a/blipr_0.2.0.tar.gz’ had non-zero exit status
    ```

it reminds us to install the `rcbc` R package first, which is also from GitHub

```r
> remotes::install_github("dirkschumacher/rcbc")
```

it throws an error due to the dependencies `cbc` (not R package)

!!! failure "requirement of system `cbc` program"
    ```bash
    Package cbc was not found in the pkg-config search path.
    Perhaps you should add the directory containing `cbc.pc'
    to the PKG_CONFIG_PATH environment variable
    Package 'cbc', required by 'virtual:world', not found
    Using PKG_CFLAGS=-I/usr/include/coin
    Using PKG_LIBS=-lCbc -lCbcSolver
    ------------------------- [ANTICONF ERROR] ----------------------------------
    Configuration failed because cbc was not found. Try installing:
    * deb: coinor-libcbc-dev, coinor-libclp-dev (Debian, Ubuntu, etc)
    * rpm: coin-or-Cbc-devel, coin-or-Clp-devel (Fedora, CentOS, RHEL)
    * brew: coin-or-tools/coinor/cbc (Mac OSX)
    If cbc is already installed, check that 'pkg-config' is in your
    PATH and PKG_CONFIG_PATH contains a cbc.pc file. If pkg-config
    is unavailable you can set INCLUDE_DIR and LIB_DIR manually via:
    R CMD INSTALL --configure-vars='INCLUDE_DIR=... LIB_DIR=...'
    ------------------------- [BEGIN ERROR MESSAGE] -----------------------------
    cat: configure.log: No such file or directory
    ------------------------- [END ERROR MESSAGE] -------------------------------
    ERROR: configuration failed for package ‘rcbc’
    * removing ‘/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/rcbc’
    Warning message:
    In i.p(...) :
    installation of package ‘/tmp/Rtmpb8HWYq/file26b46d55e8f325/rcbc_0.1.0.9001.tar.gz’ had non-zero exit status
    ```

Since we do not have `sudo` permission, and after some searching for the available packages on the HPC, it seems no the required `cbc` program.

!!! tip "install external program in conda"
    An alternative is to install `cbc` in a conda env, which does not require `sudo`.

However, if we have load an `R` env, we cannot further load `miniconda` env, 

!!! failure "cannot load miniconda after loading R"
    ```bash
    $ module load miniconda
    Lmod has detected the following error:  Cannot load module "miniconda/23.5.2" because these module(s) are loaded:
    Python

    While processing the following module(s):
        Module fullname   Module Filename
        ---------------   ---------------
        miniconda/23.5.2  /vast/palmer/apps/avx2/modules/tools/miniconda/23.5.2.lua
    ```

!!! tip "first unload Python then load miniconda"
    According to the error message, the error is due to the duplicated loading of `Python`, so we can first unload `Python`, which is loaded when loading `R`, then load `miniconda`

    ```bash
    # just need to type `Python`, then type `Tab` to autocomplete the detailed version of Python
    module unload Python/3.8.6-GCCcore-10.2.0
    module load miniconda
    ```

Now create a fresh env (or an existing env if you like) and install [`cbc` program](https://github.com/coin-or/Cbc)


```bash
conda create -n cbc coin-or-cbc
```

Then according to the above error message

> If cbc is already installed, check that 'pkg-config' is in your
    PATH and PKG_CONFIG_PATH contains a cbc.pc file. If pkg-config
    is unavailable you can set INCLUDE_DIR and LIB_DIR manually via:
    R CMD INSTALL --configure-vars='INCLUDE_DIR=... LIB_DIR=...'


!!! tip "prepend folder of `cbc.pc` to `PKG_CONFIG_PATH`"
    First determine the location of `cbc.pc`, and prepend it to `PKG_CONFIG_PATH`. It locates at

    ```bash
    $ ll | grep cbc.pc
    -rw-rw-r-- 1 lw764 zhao  398 Feb 13 12:28 cbc.pc
    $ pwd
    ~/.conda/envs/cbc/lib/pkgconfig
    ```

    Now we prepend the path to `PKG_CONFIG_PATH`

    ```bash
    export PKG_CONFIG_PATH=~/.conda/envs/cbc/lib/pkgconfig:$PKG_CONFIG_PATH
    ```


Then re-run

```r
> remotes::install_github("dirkschumacher/rcbc")
```

the previous error disappeared, although a new error comes later,

!!! failure "libCbcSolver.so.3: cannot open shared object file"
    ```r
    Error: package or namespace load failed for ‘rcbc’ in dyn.load(file, DLLpath = DLLpath, ...):
    unable to load shared object '/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/00LOCK-rcbc/00new/rcbc/libs/rcbc.so':
    libCbcSolver.so.3: cannot open shared object file: No such file or directory
    Error: loading failed
    Execution halted
    ERROR: loading failed
    * removing ‘/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/rcbc’
    Warning message:
    In i.p(...) :
    installation of package ‘/tmp/RtmppAGpHm/file270a4940b8d111/rcbc_0.1.0.9001.tar.gz’ had non-zero exit status
    ```    

!!! tip "prepend the path of `libCbcSolver.so.3` to `LD_LIBRARY_PATH`"
    This error is quite general, just need to prepend the path of `libCbcSolver.so.3` to `LD_LIBRARY_PATH`

    ```bash
    $ export LD_LIBRARY_PATH=~/.conda/envs/cbc/lib:$LD_LIBRARY_PATH
    ```

then re-run

```r
> remotes::install_github("dirkschumacher/rcbc")
```

The previous error disappers, but a new error,

!!! failure "version `GLIBCXX_3.4.29' not found"
    ```bash
    Error: package or namespace load failed for ‘rcbc’ in dyn.load(file, DLLpath = DLLpath, ...):
    unable to load shared object '/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/00LOCK-rcbc/00new/rcbc/libs/rcbc.so':
    /vast/palmer/apps/avx2/software/GCCcore/10.2.0/lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/lw764/.conda/envs/cbc/lib/libCbcSolver.so.3)
    Error: loading failed
    Execution halted
    ERROR: loading failed
    * removing ‘/vast/palmer/home.mccleary/lw764/R/x86_64-pc-linux-gnu-library/4.2/rcbc’
    Warning message:
    In i.p(...) :
    installation of package ‘/tmp/Rtmpon2JT7/file27193b6317facf/rcbc_0.1.0.9001.tar.gz’ had non-zero exit status

    ```

!!! tip "preload the correct stdc++ version"
    This is due to the conflict of `libstdc++.so.6` file. Note that `cbc` program is installed within the conda env, so it depends on the `libstdc++.so.6` file from the conda env instead of the system one `/vast/palmer/apps/avx2/software/GCCcore/10.2.0/lib64/libstdc++.so.6` mentioned in the error message,

    ```bash
    $ ldd ~/.conda/envs/cbc/lib/libCbcSolver.so.3 | grep stdc
        libstdc++.so.6 => ~/.conda/envs/cbc/lib/./libstdc++.so.6 (0x000014d306c4a000)
    ```

    To force using the `libstdc++.so.6` from the conda env, we can pre-load it before opening an R session, i.e.,

    ```bash
    $ LD_PRELOAD=/home/lw764/.conda/envs/cbc/lib/./libstdc++.so.6 R
    ```

Then `rcbc` should be successfully installed.

```bash
> remotes::install_github("dirkschumacher/rcbc")
...

** R
** inst
** byte-compile and prepare package for lazy loading
** help
*** installing help indices
** building package indices
** installing vignettes
** testing if installed package can be loaded from temporary location
** checking absolute paths in shared objects and dynamic libraries
** testing if installed package can be loaded from final location
** testing if installed package keeps a record of temporary installation path
* DONE (rcbc)
```

Now re-run to install `blipr`

```R
> remotes::install_github("amspector100/blipr")

* installing *source* package ‘blipr’ ...
** using staged installation
** R
** inst
** byte-compile and prepare package for lazy loading
** help
*** installing help indices
** building package indices
** installing vignettes
** testing if installed package can be loaded from temporary location
** testing if installed package can be loaded from final location
** testing if installed package keeps a record of temporary installation path
* DONE (blipr)
```

All is done!