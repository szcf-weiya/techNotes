# Docker

## Tutorials

- [Docker 入门教程 -- 阮一峰](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
- [Docker 微服务教程 -- 阮一峰](http://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)

## Installation

Directly type `docker` in the terminal,

```bash
$ docker

Command 'docker' not found, but can be installed with:

sudo snap install docker     # version 19.03.11, or
sudo apt  install docker.io

See 'snap info docker' for additional versions.

```

then run

```bash
sudo apt  install docker.io
```

Without permisson, it will report the following message

```bash
$ docker version
Client:
 Version:           19.03.6
 API version:       1.40
 Go version:        go1.12.17
 Git commit:        369ce74a3c
 Built:             Fri Feb 28 23:45:43 2020
 OS/Arch:           linux/amd64
 Experimental:      false
Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/version: dial unix /var/run/docker.sock: connect: permission denied
```

To [avoid permission issue](https://docs.docker.com/engine/install/linux-postinstall/),

```bash
sudo usermod -aG docker $USER
```

But it is necessary to log out and log back in to re-evaluate the group membership.

### install r via docker

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

### change the root folder

to save space, I want to [change the image installation directory:](https://stackoverflow.com/questions/24309526/how-to-change-the-docker-image-installation-directory)

```bash
$ sudo vi /etc/docker/daemon.json
{
  "data-root": "/new/path/to/docker-data"
}
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

where the [official explanation](https://docs.docker.com/engine/reference/commandline/dockerd/#daemon-configuration-file) is that

- `data-root` is the path where persisted data such as images, volumes, and cluster state are stored. The default value is /var/lib/docker

we can validate it with the `hello-world` image,

```bash
$ docker image pull hello-world
# or docker image pull library/hello-world
$ docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
hello-world         latest              d1165f221234        7 weeks ago         13.3kB
$ docker image inspect d1165
[
    {
        "Id": "sha256:d1165f2212346b2bab48cb01c1e39ee8ad1be46b87873d9ca7a4e434980a7726",
        "RepoTags": [
            "hello-world:latest"
...
        "GraphDriver": {
            "Data": {
                "MergedDir": "/media/weiya/PSSD/Programs/docker/overlay2/511d95f2c0f646ed080c006f99f8f738f967231d33aaa36a98e3e67109eb09be/merged",
                "UpperDir": "/media/weiya/PSSD/Programs/docker/overlay2/511d95f2c0f646ed080c006f99f8f738f967231d33aaa36a98e3e67109eb09be/diff",
                "WorkDir": "/media/weiya/PSSD/Programs/docker/overlay2/511d95f2c0f646ed080c006f99f8f738f967231d33aaa36a98e3e67109eb09be/work"
            },
...
```

## XAMPP

[XAMPP](https://en.wikipedia.org/wiki/XAMPP) is a free and open-source **cross-platform (X)** web server solution stack package developed by Apache Friends, consisting mainly of the **Apache HTTP Server (A)**, **MariaDB database (M)** (formerly MySQL), and interpreters for scripts written in the **PHP (P)** and **Perl (P)** programming languages.

!!! info
    [My DaSS Project](https://github.com/szcf-weiya/DaSS/)

Here is a [great docker image](https://hub.docker.com/r/tomsik68/xampp)!

Start via

```bash
#$ docker pull tomsik68/xampp
$ docker run --name myXAMPP -p 41061:22 -p 41062:80 -d -v ~/my_web_pages:/www tomsik68/xampp:8
```

!!! tip
    - `docker run` and `docker container run` are exactly the same [:material-stack-overflow:](https://stackoverflow.com/questions/51247609/what-is-the-difference-between-docker-run-and-docker-container-run)
    - since `docker run` will automatically download the image if no installed, then `docker pull` is unnecessary. [:link:](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
    - `-v /HOST-DIR:/CONTAINER-DIR` creates a bind mount.
    - `-p hostPort:containerPort` publishes the container's port to the host.
    - `-d` runs the container in the background and print the new container ID.
    - :key: More details can be checked via `man docker-run`.

Then we can see the container via 

```bash
$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                                                    NAMES
43e5a49cbfd5        tomsik68/xampp      "sh /startup.sh"    18 seconds ago      Up 17 seconds       3306/tcp, 0.0.0.0:41061->22/tcp, 0.0.0.0:41062->80/tcp   myXAMPP
```

Stop via

```bash
#$ docker container stop/kill [containerID]
$ docker stop/kill [containerID]
# then
$ docker stop 43e5
43e5
$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

!!! tip
    - similarly, `docker container stop/kill` can be abbreviated as `docker stop/kill`
    - `kill` 向容器里面的主进程发出 SIGKILL 信号，而 `stop` 发出 SIGTERM 信号，然后过一段时间再发出 SIGKILL 信号。两者差异是，应用程序收到 SIGTERM 信号后，可以自行进行收尾清理工作，但也可以不理会这个信号。如果收到 SIGKILL 信号，就会强行立即终止，那些正在进行的操作会全部丢失。[:link:](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
    - `containerID` 无需写全，只要能区分即可
    - since we have specified the name via `--name myXAMPP`, we can replace the containerID with such name.

Restart via

```bash
# find the container ID
$ docker container ls -a
$ docker container start [containerID]/[containerNAME]
```

!!! tip
    - `docker container ls` only shows the running ones, but `-a` will show all containers. More details can be found in `man docker-container-ls`

Establish a ssh connection,

```bash
$ ssh root@127.0.0.1 -p 41061
```

it sounds like the port-forwarding if we view the container as another linux machine.

!!! info
    Both default username and password are `root`.

Alternatively, we can get a shell terminal insider the container, just like ssh,

```bash
$ docker exec -it myXAMPP bash
```

!!! tip
    - `-t` allocates a pseudo-TTY.
    - `-i` keeps STDIN open even if not attached.
    - `docker [container] exec` 用于进入一个正在运行的 container. 如果 `docker run` 命令运行容器时，没有使用 `-it`，则需要这个命令进入容器。[:link:](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)

If we are inside the container, we can export the path to use the commands provided by XAMPP,

```bash
# inside docker container
export PATH=/opt/lampp/bin:$PATH
# or add it to `.bashrc` of the container
```

If we modified the configuration of XAMPP, we need to restart the Apache server via

```bash
docker exec myXAMPP /opt/lampp/lampp restart
```

## Python (for a non-internet env)

First of all, write a dockerfile

```bash
cat Dockerfile
FROM python:3.7
RUN pip install jieba
```

then build it with

```bash
$ docker image build -t py37jieba:0.0.1 .
```

and test it locally with

```bash
$ docker run -it py37jieba:0.0.1 
Python 3.7.10 (default, May 12 2021, 16:05:48) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jieba
>>> jieba.cut("他来到了网易杭研大厦")
<generator object Tokenizer.cut at 0x7fda1c981bd0>
>>> print(", ".join(jieba.cut("他来到了网易杭研大厦")))
Building prefix dict from the default dictionary ...
Dumping model to file cache /tmp/jieba.cache
Loading model cost 0.924 seconds.
Prefix dict has been built successfully.
他, 来到, 了, 网易, 杭研, 大厦
```

save the image with

```bash
$ docker save py37jieba:0.0.1 | gzip > py37jieba-0.0.1.tar.gz
```

refer to [hubutui/docker-for-env-without-internet-access](https://github.com/hubutui/docker-for-env-without-internet-access)