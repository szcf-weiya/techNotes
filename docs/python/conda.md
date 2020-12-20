# Conda


## 正式认识conda

参考[https://conda.io/docs/user-guide/getting-started.html](https://conda.io/docs/user-guide/getting-started.html)

## conda 在window下设置

win10下已经装了anaconda，spyder(2.7)，现想再装上python3，于是利用conda创建一个python3的环境`bunny`。

```cmd
conda create --name bunny python=3
```

切换到bunny环境

```cmd
activate bunny
```

参考[spyder3 doc](https://pythonhosted.org/spyder/installation.html)

```cmd
pip install spyder3
```

## 为 py3 安装 spyder

1. 先建一个conda环境bunny，安装python3.4，因为要支持pyside，而经试验3.5+不支持。
2. 安装cmake
3. pyside出现keyerrror

转向py3.6
不装pyside，而装pyqt5
```bash
pip install pyqt5
```

最终在新建的另一个 conda 环境 snakes 中装好了 Python3.6 及 spyder3，通过下面命令运行就 OK 了。

```bash
source activate snakes
spyder3 &
```

## disable default `base`

```bash
conda config --set auto_activate_base false
```

refer to [How do I prevent Conda from activating the base environment by default?](https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default)

actually, this is equivalent to add a line in `~/.condarc`

```bash
auto_activate_base: false
```

## conda package 的版本号不匹配

在用 anaconda3 新建 py37 环境后，安装 spyder，但是运行时弹出

![](spyder-qtconsole.png)

而用 `conda list | grep qtconsole` 检查发现

```bash
qtconsole                 4.6.0                      py_0
```

隐隐约约感觉是装了多个版本，想卸载掉错误的版本，但都没找到 qtconsole 怎么查看版本号。然后 Google 发现另外一个类似的问题，

[Getting wrong version of packages using Conda](https://stackoverflow.com/questions/55350956/getting-wrong-version-of-packages-using-conda)

于是我也去检查了 ipython 的版本，发现 

```bash
$ ipython --version
7.7.0
```

而

```bash
$ conda list | grep ipython
ipython                   7.10.2           py37h39e3cac_0
```

同样存在版本号不一致的问题。

所以按照评论的建议，用 

```bash
pip uninstall ipython
```

首先解决了 ipython 的版本号不一致的问题。

![](ipython.png)

受此启发，用

```bash
pip uninstall qtconsole
```

解决了 qtconsole 的问题。

## conda 指定 env 路径

如果直接在创建时通过 `-p` 指定路径

```bash
conda create -p ... python=x.x
```

注意如果指定路径，则不需要 `--name`， 因为默认会将路径最后的文件名看成是 env 的 name。

则 activate 的时候需要加上整个路径。

在创建之前可以先在 [`.condarc` 中的 `env_dirs`](https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs) 项下添加指定的路径。

## conda 迁移环境

[官方文档](https://www.anaconda.com/blog/moving-conda-environments)介绍了四种环境迁移的方式，

- clone
- Spec List
- Environment.yml
- Conda Pack

其中第四种似乎更符合需求，因为第二三种需要重新下载 package，而第一种不太能直接（额外指定参数或许可以）支持多个 envs folder 间的切换，但是第四种本质上就是打包解压，所以何不如直接 `mv` 移动呢，毕竟我是在同一台电脑上操作，试了一下，果然成功了。

```bash
$ conda env list
# conda environments:
#
base                  *  /home/weiya/anaconda3
py27                     /home/weiya/anaconda3/envs/py27
py35                     /media/weiya/PSSD/Programs/anaconda3/envs/py35

$ mv anaconda3/envs/py27/ /media/weiya/PSSD/Programs/anaconda3/envs/
$ conda env list
# conda environments:
#
base                  *  /home/weiya/anaconda3
py27                     /media/weiya/PSSD/Programs/anaconda3/envs/py27
py35                     /media/weiya/PSSD/Programs/anaconda3/envs/py35
```

但是发现了个小问题，`pip`用不了，比如

```bash
$ pip install pymdown-extensions
bash: /media/weiya/PSSD/Programs/anaconda3/envs/py36/bin/pip: /home/weiya/anaconda3/envs/py36/bin/python: bad interpreter: No such file or directory
```

其还是想调用原先路径下的 python，然后重新装一下 `conda install pip` 就好了。

## conda clean

笔记本硬盘余量告急，然后发现 `anaconda` 文件夹下竟然有 15GB，所以想有没有什么方法清理一下，果然有 [`conda clean`](https://docs.conda.io/projects/conda/en/latest/commands/clean.html) 这句命令，

> Remove unused packages and caches.

但是有点担心其 `unused` 的定义，是多长时间没有用过吗？比如一个月之类的，如果是这样，意义并不大。后来找到了这个[回答](https://stackoverflow.com/questions/51960539/where-does-conda-clean-remove-packages-from)

> An "unused" package is one that's not used in any environment.

以及 [Conda clean 净化Anaconda](https://www.jianshu.com/p/f14ac62bef99)

> - 有些包安装之后，从来没有使用过；
> - 一些安装包的tar包也保留在了计算机中；
> - 由于依赖或者环境等原因，某些包的不同版本重复安装。

于是比较放心地运行了 `conda clean -a`，一下子清理出 8.6G 的空间。

## 不同 environment 的 jupyter

其实不用对每个 environment 安装单独的 jupyter，只有安装好 ipykernel 就好，这样都能从 base environment 中通过 jupyter 来选择不同 kernel，详见 [Kernels for different environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments)

### 安装 julia 的 kernel

```julia
> add IJulia
```
