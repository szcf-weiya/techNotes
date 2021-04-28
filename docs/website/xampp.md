# XAMPP

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