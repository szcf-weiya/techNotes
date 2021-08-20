# Installation Notes of Samtools

by Sheng Lian (20210818)

## Introduction

- Official Site: [http://www.htslib.org/](http://www.htslib.org/)

## Installation without root

- First install several libraries, e.g., zlib, bzip2, ... 

  (Basically follow the instruction [here](https://www.jianshu.com/p/da92ca36a220).)

  ```bash
  tar xvf samtools-1.13.tar.bz2
  cd samtools-1.13
  ./configure --prefix=$HOME/usr
  make
  make install
  ```

- To solve the problem in `./configure` results: 'No such file or directory' when installing bzip2: 
    
  - Download from [SourceForge](https://sourceforge.net/projects/bzip2/) and `tar -zxvf bzip2-1.0.6.tar.gz`;
  
  - Add Path (`bzip2-1.0.6`): 
    
    ```bash
    export CPPFLAGS="-I$HOME/usr/include -I$HOME/bzip2-1.0.6"
    export LDFLAGS="-L$HOME/usr/lib -L$HOME/usr/lib64 -L$HOME/bzip2-1.0.6"
    export LD_LIBRARY_PATH=$HOME/usr/lib:$HOME/usr/lib64:$HOME/bzip2-1.0.6
    ```

## Using Conda 
  
- Install Miniconda: 

  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh
  ```

- `conda install -c bioconda samtools`
