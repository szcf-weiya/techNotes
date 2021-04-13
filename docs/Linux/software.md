# Big Softwares on Linux

## Atom

### atom 自动更新

[atom](https://launchpad.net/~webupd8team/+archive/ubuntu/atom/)

```
sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
```

### proxy 设置

Atom 的包管理器是 [apm](https://github.com/atom/apm#behind-a-firewall)，其中有介绍怎么设置 proxy，即

```bash
apm config set strict-ssl false
apm config set http-proxy http://127.0.0.1:8118
```

### spell-check

By default, one of the core package [spell-check](https://github.com/atom/spell-check) does not check `.tex` file, although there is [another package](https://github.com/AtomLinter/linter-spell-latex) for checking the spelling in `.tex`, it does not work and not recently update, then I investigate the `spell-check` package to add the support for `.tex`. Actually, it is quite easy, just to add the scope of the `.tex` file, which can be found by `Editor: Log Cursor Scope`. So I add `text.tex.latex`, but it would be annoying to highlighter the native latex command, such as `\newcommand`, then I found that there is a `Excluded Scopes` in the config page of `spell-check`, so we only need to add the scope name of such native latex command, which again can be found by `Editor: Log Cursor Scope` if we put the cursor on the line of the commands. Finally, I add

```bash
meta.preamble.latex, punctuation.definition.arguments.end.latex, support.function.general.tex, support.type.function.other.latex, storage.type.function.latex, markup.underline.link.https.hyperlink
```

to the whitelist, each of which is identified by tries, such as cannot continue to add `meta.group.braces.tex` since the original text would also cannot be checked.

### Soft wrap

Soft wrap is proper for `.tex` file, or `.md` file, but not necessary for the programming file. We can turn off the soft wrap globally in `Setting > Editor`, and actually we can reset it for each language, which can be toggled in `Setting > Package > language-<language name> > Soft Wrap`.

refer to [Toggle Soft Wrap by Default?](https://discuss.atom.io/t/toggle-soft-wrap-by-default/58911/5)

### minimap

装好用了一晚上，但是第二天用的时候却怎么也打不开了，尝试设置 Key binding，即便已经设置为了自动启动，所以原因并不是这个。

后来通过 `apm` 安装低版本便成功了！

![](minimap.png)

### terminal

之前一直在使用 `Platformio Ide Terminal v2.10.1`, 但是最近一段时间经常打不开 terminal，后来在其 repo issue 中看到类似的问题，然后有人指出这个 package 其实[不再维护](https://github.com/platformio/platformio-atom-ide-terminal/issues/543)，并且推荐了

- terminus: https://github.com/bus-stop/terminus
- x-terminal: https://github.com/bus-stop/x-terminal

打不开 terminal 的原因应该与下文中提到的 VS code 类似，在替换自动启动方式之前，试过在 x-terminal 中启动程序 `/bin/bash` 添加 `--noprofile` 选项，但是报错，于是直接选择了 terminus.

## 百度网盘

发现百度网盘出了 Linux 版，但是在 Ubuntu 16.04 似乎运行不了——能下载安装但是无法打开运行。

目前版本为 Linux版 V2.0.2（更新时间：2019-07-25）

[官网](https://pan.baidu.com/download)显示目前只支持

> 适应系统：中标麒麟桌面操作系统软件（兆芯版） V7.0、Ubuntu V18.04

于是寻找替代方案。

### bcloud

项目地址：[https://github.com/XuShaohua/bcloud](https://github.com/XuShaohua/bcloud)

但是四五年没有更新了。安装试了一下，登录不了，遂放弃。

### PanDownload

[https://www.baiduwp.com](https://www.baiduwp.com)

不需要安装客户端，只需要输入网盘分享链接和提取码，便可以下载文件（而百度网盘本身下载文件需要打开客户端）。不过速度似乎不咋地

### bypy

逛到了另外一个客户端，项目地址：[https://github.com/houtianze/bypy](https://github.com/houtianze/bypy)

还挺活跃，五个月前有更新。

测试了一下，相当于在网盘内新建了 `/apps/bypy` 文件夹，然后可以同步该文件夹内的内容，似乎不能直接对文件夹外的文件进行操作。尽管这样，也是很好的了，以后文件可以存放在这个文件夹下。

当然，还是期待官网本身支持。

常用命令：

```bash
bypy syncup
bypy syncdown
```

## Chrome

### 黑屏

参考 [chrome黑屏解决](https://blog.csdn.net/jjddrushi/article/details/79155421)   

进入休眠状态后，睡了一晚上，第二天早上打开 chrome 便黑屏了，然后采用

```bash
chrome -disable-gpu
```

再设定

```
Use hardware acceleration when available
```
false，再点击 relaunch，则黑屏的页面都回来了，不需要重启整个 chrome。

### 沙盒 sandbox

在 Julia 中使用 Plotly 画图时，报出

```julia
[0531/160811.236665:ERROR:nacl_helper_linux.cc(308)] NaCl helper process running without a sandbox!
Most likely you need to configure your SUID sandbox correctly
```

通过 `ps -aef | grep chrome` 查看 chrome 的运行参数，没有发现 `no-sandbox`，即默认应该是开启的，所以现在不清楚了。

## Docker

### Tutorials

- [Docker 入门教程 -- 阮一峰](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
- [Docker 微服务教程 -- 阮一峰](http://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)

### Installation

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

[change the image installation directory:](https://stackoverflow.com/questions/24309526/how-to-change-the-docker-image-installation-directory)

```bash
$ sudo vi /etc/docker/daemon.json
{
  "data-root": "/new/path/to/docker-data"
}
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

## Emacs

### 常用命令

1. 切换缓存区：C-o
2. 水平新建缓存区：C-2
3. 垂直新建缓存区：C-3
4. 关闭当前缓存区：C-0
5. 删除缓存区：C-k
6. 只保留当前缓存区：C-1

### Emacs使用Fcitx中文

参考博客：[fcitx-emacs](http://wangzhe3224.github.io/emacs/2015/08/31/fcitx-emacs.html)

- Step 1: 确定系统当前支持的字符集

```bash
locale -a
```

若其中有 zh_CN.utf8，则表明已经包含了中文字符集。

- Step 2: 设置系统变量

```bash
emacs ~/.bashrc
export LC_CTYPE=zh_CN.utf8 
source ~/.bashrc
```

- 配置文件: [http://download.csdn.net/download/karotte/3812760](http://download.csdn.net/download/karotte/3812760)
- 自动补全: 参考[emacs自动补全插件auto-complet和yasnippet，安装、配置和扩展](http://www.cnblogs.com/liyongmou/archive/2013/04/26/3044155.html#sec-1-2)

## ImageMagick

### Add HEIC support in ImageMagick

!!! fail
    failed.

上次从源码按安装了 ImageMagick 7.0.10-6，刚刚又看到可以[添加对 HEIC 格式的支持](https://askubuntu.com/questions/958355/any-app-on-ubuntu-to-open-and-or-convert-heif-pictures-heic-high-efficiency-i)，于是准备重新编译安装

```bash
$ ./configure --with-modules --with-libheif
...
               Option                        Value
------------------------------------------------------------------------------
...
Delegate library configuration:
...
  HEIC              --with-heic=yes             no
```

跟 HEIC 似乎只有这一条，但其实如果去掉 `--with-libheif`，结果并不会有变化，后来发现这个选项其实并没有正确识别，

```bash
configure: WARNING: unrecognized options: --with-libheif
configure:
```

然后试着

```bash
sudo apt-get install libheif1
```

但最后一列还是 no，然后再试着

```bash
sudo apt-get install libheif-dev
```

最后一列终于变成 yes 了。于是继续 `make`，然而却报出了 bug

```bash
coders/heic.c: In function ‘ReadHEICColorProfile’:
coders/heic.c:143:5: warning: unused variable ‘length’ [-Wunused-variable]
     length;
     ^~~~~~
coders/heic.c: In function ‘ReadHEICImage’:
coders/heic.c:452:9: warning: implicit declaration of function ‘heif_context_read_from_memory_without_copy’; did you mean ‘heif_context_read_from_memory’? [-Wimplicit-function-declaration]
   error=heif_context_read_from_memory_without_copy(heif_context,file_data,
         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         heif_context_read_from_memory
coders/heic.c:452:8: error: incompatible types when assigning to type ‘struct heif_error’ from type ‘int’
   error=heif_context_read_from_memory_without_copy(heif_context,file_data,
        ^
At top level:
coders/heic.c:94:3: warning: ‘xmp_namespace’ defined but not used [-Wunused-const-variable=]
   xmp_namespace[] = "http://ns.adobe.com/xap/1.0/ ";
   ^~~~~~~~~~~~~
Makefile:10388: recipe for target 'coders/heic_la-heic.lo' failed
make[1]: *** [coders/heic_la-heic.lo] Error 1
make[1]: Leaving directory '/home/weiya/src/ImageMagick-7.0.10-6'
Makefile:5988: recipe for target 'all' failed
make: *** [all] Error 2
```

## Input Methods for Chinese

!!! info
    目前试用 ibus-rime……

### fcitx-sougou

需要 fcitx，若没有装，

```bash
sudo apt-get install fcitx-bin
sudo apt-get install fcitx-table
```

然后将输入法切换成 fcitx，在设置中语言那里，

最后下载按照搜狗输入法，安装时我出现这样的问题导致安装失败，

> No such key 'Gtk/IMModule' in schema 'org.gnome.settings-daemon.plugins.xsettings' as specified in override file '/usr/share/glib-2.0/schemas/50_sogoupinyin.gschema.override'; ignoring override for this key.

参考 [Install sogoupinyin on ubuntu 16.04LTS, with error 'Gtk/IMModule'](https://askubuntu.com/questions/883506/install-sogoupinyin-on-ubuntu-16-04lts-with-error-gtk-immodule)，将 `/usr/share/glib-2.0/schemas/50_sogoupinyin.gschema.override` 中的 `IMModule` 一行改成

```bash
overrides={'Gtk/IMModule':<'fcitx'>}
```

然后再运行

```bash
sudo glib-compile-schemas /usr/share/glib-2.0/schemas/
```

再次安装便成功了。

最后在语言栏中添加搜狗拼音的输入法即可。

参考 [解决Ubuntu 18.04中文输入法的问题，安装搜狗拼音](https://blog.csdn.net/fx_yzjy101/article/details/80243710)

### fcitx-googlepinyin

因为最近发现 wechat 崩溃时经常是无法切换到中文输入，所以怀疑会不会是搜狗输入法引起的，于是想尝试不同的中文输入法。在知乎上看到这个问题，[Ubuntu 上最好用的中文输入法是什么？](https://www.zhihu.com/question/19839748)

里面也有人反映搜狗输入法会导致其他程序崩溃，

> 搜狗输入法导致Jetbrains全家桶崩溃 @口袋里的星辰
> ubuntu用搜狗输入法，会出各种问题，有其实和linux版微信一起用，坑就更多了。。@南瓜派三蔬

于是更加坚定尝试其他输入法，因为很多人都推荐了谷歌拼音，便首先尝试了它。因为此前在装搜狗的时候已经将 ibus 换成了 fcitx，所以后面只需要添加新的输入法即可，

```bash
$ sudo apt install fcitx-googlepinyin
```

然后重启输入法（无需重启电脑），再打开输入法配置界面，此时不出意外已经有 google pinyin 的选项。因为一种输入法就足够了，而且避免后台影响，取消了搜狗输入法（并没有卸载程序）。

试用了一下，还行，只是界面略微有点丑，不过这也不是重点。

### fcitx-baidu

既然添加个输入法这么简单，那索性再试试其它的，百度输入法可以在其官网中下载的到 `.deb` 文件，然后安装并重启输入法。

正如上述知乎回答提到的，它乱码了！

### fcitx-rime

这主要是繁体中文，不过似乎应该也能切换简体。本身这是基于 ibus 的，不过 [fcitx 团队](https://github.com/fcitx)有在维护 fcitx 的版本，

```bash
$ sudo apt install fcitx-rime
```

因为想同时比较其与谷歌拼音的体验，所以目前同时保留了这两个输入法，可以通过 `SHIFT+CTRL` 快速切换输入法。

RIME 默认是繁体的，可以通过 `` CTRL+` `` 来切换简繁体，另外也有全半角等设置。

!!! note
    除了这些在 fcitx4 上的方案，也许过段时间会尝试[更新的输入法框架 fcitx5](https://www.zhihu.com/question/333951476)

虽然谷歌拼音和fcitx-rime都表现得不错，但是默认的 UI 实在有点丑，看到 [kimpanel](https://fcitx-im.org/wiki/Kimpanel) 会比较好看，想试一试，采用 gnome-shell 安装，但是竟然 [no popup window](https://github.com/wengxt/gnome-shell-extension-kimpanel/issues/53)，虽放弃。

### ibus-rime

虽然 kimpanel 行不通，但是 [wengxt](https://github.com/wengxt/gnome-shell-extension-kimpanel/issues/53#issuecomment-812811613) 的回答让我意识到兴许 ibus 可以试一试。

最开始用 ubuntu 的时候用过一段时间 ibus，那时应该还是在 14.04 的机子上，确实不太好用，后来一直换成了 fcitx。不过现在版本已经 18.04 了，兴许会好点，而且比较喜欢这种 ui 跟系统很配的感觉。

![](https://user-images.githubusercontent.com/13688320/113474286-b60ec980-94a1-11eb-84b3-191bf1940f82.png)

但是明显感觉连续功能确实还不跟够好。另外有个问题是，在浏览器或者 libreoffice 中输入时，备选框总在左下角，[解决方案](https://askubuntu.com/questions/549900/ibs-chinese-pinyin-input-candidates-appear-at-bottom-of-screen)为

```bash
$ sudo apt install ibus-gtk
```

因为 rime 本身就是 ibus 的，而且在 fcitx 环境下体验效果确实不错，所以想试试 [ibus-rime](https://github.com/rime/home/wiki/RimeWithIBus)。

```bash
$ sudo apt-get install ibus-rime
```

因为 ibus 是系统默认的，所以其不像 fcitx 有单独的配置框，而是直接在系统设置的 “Region & Language” 中进行设置，添加 “Input Source” 即可。

## Kazam

Ubuntu 下 kazam 录屏 没声音解决方案

[http://www.cnblogs.com/xn--gzr/p/6195317.html](http://www.cnblogs.com/xn--gzr/p/6195317.html)

### Kazam video format

cannot open in window

solution

```
ffmpeg -i in.mp4 -pix_fmt yuv420p -c:a copy -movflags +faststart out.mp4
```

refer to [convert KAZAM video file to a file, playable in windows media player](https://video.stackexchange.com/questions/20162/convert-kazam-video-file-to-a-file-playable-in-windows-media-player)

## Octave

参考[Octave for Debian systems](http://wiki.octave.org/Octave_for_Debian_systems)

另外帮助文档见[GNU Octave](https://www.gnu.org/software/octave/doc/interpreter/)

## Okular

当初使用 Ubuntu 16.04 时，Okular 是通过 snap 安装的，可能参考了[这个](https://askubuntu.com/questions/976248/how-to-install-latest-version-of-okular-on-ubuntu-16-04)?

```bash
sudo snap install okular
```

但是更新到 Ubuntu 18.04 后，发现在移动硬盘的文档打不开，而之前没碰到过这样的问题，一开始还以为是移动硬盘命名问题，之前曾经碰到过某个程序（忘记了）不允许路径存在空格，而移动硬盘默认名字有空格，于是曾经更改过名字（忘记了怎么更改）。原本以为可能更新系统使得这个更改失效了，还想着再找找怎么更改，但是找到一堆怎么更改卷标名的，最后才发现路径中名字确实应该更改成功了。

所以问题还是回到 okular 本身，通过 snap 和 apt 安装是两个不同的版本，图标也有点差异，然后发现也有人跟我有[同样的问题](https://askubuntu.com/questions/1137830/cannot-open-pdf-files-in-mounted-usb-drive-using-okular)，有人回复说

> Okular does not support removable media while installed as Snap.

于是卸掉 snap 版的 okular，转而安装 apt 版本的，

```bash
sudo apt-get install okular
```

类似地，[通过 snap 安装的 `gimp`](https://askubuntu.com/questions/958355/any-app-on-ubuntu-to-open-and-or-convert-heif-pictures-heic-high-efficiency-i) 不能打开移动硬盘中的文件，但是如果换成 apt-get 安装的，则又不支持 `.heic` 文件格式。

发现有些图标不能正常显示，网上也找到了类似的问题，

- [KDE application icon not displayed in Ubuntu](https://askubuntu.com/questions/1007563/kde-application-icon-not-displayed-in-ubuntu)
- [State of Okular in Ubuntu 17.10?](https://askubuntu.com/questions/999551/state-of-okular-in-ubuntu-17-10)

尝试了其中的解决方案，但均未成功，最后的解决方案是 [navigation panel icons missing on standard install of 17.04](https://bugs.launchpad.net/ubuntu/+source/okular/+bug/1698656)

```bash
As a workaround, what worked for me was:

$ sudo apt install systemsettings kde-config-gtk-style kde-config-gtk-style-preview oxygen-icon-theme

* systemsettings for the app systemsettings5;

* kde-config-gtk-style enables the Appearance module in systemsettings5;

* kde-config-gtk-style-preview allows previewing the themes without restarting the GTK applications;

* oxygen-icon-theme is an alternative theme to use in KDE applications.

Then, run systemsettings5, click on Application Style, select Oxygen as a Fallback theme, click on Apply.
```

最后我的配置是

![](okular-icons.png)

可以尝试不同配置，因为刚开始打开的，似乎并不是之前系统的配置。

### latex in annotation

okular 的 note 功能支持 LaTeX，当输入 `$$...$$` 时会提示要不要转换为 latex，点击后但是报错，

```bash
latex is not executable
```

注意到 `latex` 的 PATH 是定义在 `.bashrc` 中，而通过 zotero 调用 okular 时并不会 source `.bashrc`，只有通过 bash shell 调用的程序采用 source 到 `.bashrc`，也就是在终端中调用 okular 时，latex 显示正常。

研究图形界面程序调用 path 的机制似乎是一种解决方案，但觉得可能过于复杂，其实之前在 atom 中也出现过类似的问题。可能的方案是在 `.profile` 中添加 PATH，可能有用的[参考博客](https://medium.com/@abhinavkorpal/bash-profile-vs-bashrc-c52534a787d3)。

于是我采用更简单的方案，在 `/usr/bin` 中添加 `latex` 的 soft link，添加后报了新错，

```bash
dvipng is not executable
```

但至少证明这条思路是可行的，于是继续添加 `dvipng` 的 soft link，最后解决了问题！

### 自定义签名

可以通过 `stamp` 功能自定义签名，首先准备好签名图片，然后保存到某个文件夹，比如 `~/.kde/share/icons/signature.png`，然后进入 stamp 的配置界面，下拉框中直接输入签名图片所在的路径。参考 [How to add a Signature stamp to Okular](https://askubuntu.com/questions/1132658/how-to-add-a-signature-stamp-to-okular)

但是并不能存为 pdf，或者被其他软件看到，用 Acrobat 打开会有个打叉的部分，但是看不到签名，[已经被标记为 bug，但似乎还未解决](https://bugs.launchpad.net/ubuntu/+source/okular/+bug/1859632)。

## Onedrive

### first try

[xybu/onedrive-d-old](https://github.com/xybu/onedrive-d-old), but doesn't support exchange account.

### second try

[skilion/onedrive](https://github.com/skilion/onedrive), perfect!

note that the automatic monitor would occupy much CPU, the service can be disable or enable by the following command,

```bash
~$ systemctl --user disable onedrive
Removed /home/weiya/.config/systemd/user/default.target.wants/onedrive.service.
~$ systemctl --user enable onedrive
Created symlink /home/weiya/.config/systemd/user/default.target.wants/onedrive.service → /usr/lib/systemd/user/onedrive.service.
```

but it seems that we also need

```bash
systemctl --user start onedrive
systemctl --user stop onedrive
```

### Change to [abraunegg/onedrive](https://github.com/abraunegg/onedrive)

I found that it will auto run after startup, actually with [skilion/onedrive](https://github.com/skilion/onedrive), sometimes it also starts automatically. Then I tried

```bash
$ sudo systemctl disable onedrive.service
Failed to disable unit: Unit file onedrive.service does not exist.
```

and then I note that [OneDrive service running as a non-root user via systemd (with notifications enabled) (Arch, Ubuntu, Debian, OpenSuSE, Fedora)](https://github.com/abraunegg/onedrive/blob/master/docs/USAGE.md#onedrive-service-running-as-a-non-root-user-via-systemd-with-notifications-enabled-arch-ubuntu-debian-opensuse-fedora)

then I tried

```bash
$ sudo systemctl disable onedrive@weiya.service
```

no error.

Then I also tried

```bash
$ systemctl --user disable onedrive
Removed /home/weiya/.config/systemd/user/default.target.wants/onedrive.service.
```

It seems OK now, and pay attention to the difference of the above similar commands.

## Peek

[homepage](https://github.com/phw/peek), easy to use, can convert to gif.

## Rhythmbox

右键似乎可以修改歌曲的 properties，其中包括 artist，album，但是却不能编辑，然后[查了一下](https://askubuntu.com/questions/612711/rhythmbox-cannot-edit-properties)，是权限问题，

```bash
chmod u+w CloudMusic/ -R
```

where more details about `u+w` can be found in the manual.

```bash
$ man chmod

The format of a symbolic mode is [ugoa...][[-+=][perms...]...], where perms  is
either  zero  or  more letters from the set rwxXst, or a single letter from the
set ugo.  Multiple symbolic modes can be given, separated by commas.

A combination of the letters ugoa controls which users' access to the file will
be  changed:  the  user  who  owns it (u), other users in the file's group (g),
other users not in the file's group (o), or all users (a).  If  none  of  these
are  given,  the  effect  is as if (a) were given, but bits that are set in the
umask are not affected.
```

但是对 `.wav` 文件仍不能编辑 properties，后来才知道应该是 wav 不支持 tag

> [WAVs don't have tags. Trying to force them to have tags will cause them to not work as WAVs any more. Convert them to FLAC files first (which does support tags) and then tag them.](https://ubuntuforums.org/showthread.php?p=7817206)

但是还是有方法来修改的，比如 [How do I edit a metadata in a WAV file?](https://www.quora.com/How-do-I-edit-a-metadata-in-a-WAV-file)

于是我尝试了 [kid3](https://kid3.kde.org/)

```bash
sudo add-apt-repository ppa:ufleisch/kid3
sudo apt-get update
sudo apt-get install kid3     # KDE users
```

一开始觉得 `kid3-cli` 足够了，但是试了一下感觉学习成本太高，索性换回 kde 版本的。

但是似乎在 `kid3` 中修改完并没有信息，只是会把删去的 genre 信息变为 unknown。

## Rstudio

### Rstudio 不能切换中文输入（fctix）

参考[Rstudio 不能切换中文输入（fctix）](http://blog.csdn.net/qq_27755195/article/details/51002620)

- [Ubuntu 16.04 + Fcitx + RStudio 1.0で日本語を入力する方法](http://blog.goo.ne.jp/ikunya/e/8508d21055503d0560efc245aa787831)
- [Using RStudio 0.99 with Fctix on Linux](https://support.rstudio.com/hc/en-us/articles/205605748-Using-RStudio-0-99-with-Fctix-on-Linux)

曾经按照上述的指导能够解决这个问题，即将系统的 qt5 的 `libfcitxplatforminputcontextplugin.so` 手动添加到 rstudio 安装目录下的 plugins 中，即

```bash
sudo ln -s /usr/lib/$(dpkg-architecture -qDEB_BUILD_MULTIARCH)/qt5/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so /usr/lib/rstudio/bin/plugins/platforminputcontexts/
```

但是后来又失败了，猜测原因可能是 qt5 的版本不再兼容了。在 Rstudio 顶部的菜单栏中，点击 Help > About Rstudio 可以找到具体的 qt 版本信息，比如 RStudio (Version 1.2.5001) 依赖 QtWebEngine/5.12.1，而系统的 Qt 插件版本没那么高，所以也能理解 `libfcitxplatforminputcontextplugin.so` 为什么不再有用了。一种解决方案便是手动重新编译与 Rstudio 中匹配的 Qt 插件的版本，但是似乎比较繁琐，而且也不能一劳永逸，如果 rstudio 更新，还是会失效。

索性不折腾了。如果真的需要中文，就用其他编辑器吧。期待 rstudio 官方早日解决这个问题……

### 更新rstudio 后闪退

安装 rstudio 应该采用

```
sudo apt-get install gdebi-core
wget https://download1.rstudio.org/rstudio-1.0.44-amd64.deb
sudo gdebi rstudio-1.0.44-amd64.deb
```

而非
```
sudo dpkg -i
```

另外，如果不行，删除后再装
```
sudo apt-get remove rstudio
```

## TeXLive

### Install TeXLive 2020

The texlive2017 for Ubuntu cannot work for me, it reports

> fatal: Could not undump 6994 4-byte item(s) ...

and try

> fmtutil-sys --all

but does not work, refer to [Error Message: “tex: fatal: Could not undump 1 4-byte item(s) from”](https://tex.stackexchange.com/questions/141838/error-message-tex-fatal-could-not-undump-1-4-byte-items-from), but does not work.

And I also try uninstall and reinstall texlive, but it still does not work.

Then finally I decided to install the latest TeXLive 2020, [TeX Live - Quick install](https://tug.org/texlive/quickinstall.html), follow the instructions, but note that the mirror url should append `path/systems/texlive/tlnet`.

```bash
install-tl --location http://mirror.example.org/ctan/path/systems/texlive/tlnet
```

And note that the [steps for compeletely removing the installed TeXLive](https://tex.stackexchange.com/questions/95483/how-to-remove-everything-related-to-tex-live-for-fresh-install-on-ubuntu).

If without root privilege, when running `install-tl`, type `D` to change the directory, and actually changing the first `<1>` would change all other directories.

## Thunderbird

- 添加学校邮箱时，必须采用学号形式的邮箱，不要用 alias 形式的，alias 验证会出问题。

### Upgrade 68 to 78

最近，学校强制要求使用 2FA，但是根据之前短暂的使用经验，2FA 对邮箱客户端的支持很有限，比如就不支持 Ubuntu 系统上的 thunderbird，所以那次用完之后立马发邮件申请注销（因为 2FA 一旦设定自己无法取消）。

不过幸运的是，看到最近新版本的 thunderbird 支持 Oauth2，比如 [Thunderbird 77 supports IMAP using OAuth2 on Office 365. See https://bugzilla.mozilla.org/show_bug.cgi?id=1528136 for more details.](https://techcommunity.microsoft.com/t5/exchange-team-blog/announcing-oauth-2-0-support-for-imap-and-smtp-auth-protocols-in/bc-p/1354695/highlight/true#M28183) [Office 365 (Thunderbird) - Configure Modern Authentication](https://kb.wisc.edu/helpdesk/page.php?id=102005)

所以准备试试下载新版本，因为似乎不能直接简单的 upgrade 升级到 78.

一开始填错了服务器，漏掉了 `outlook.office365.com` 中的 `365`，还以为 78 不行，毕竟前面两个给的链接说是 77beta，万一 beta 的功能又被砍掉了呢：（后来发现是杞人忧天

将验证方法改为 `Oauth2`，然后重启便跳出了熟悉的登录界面，大功告成！

不过有个问题是，这两个版本是同时存在的，profile 是不共用的，我如果要用 78，还需要把其他邮箱重新设置一遍，已经订阅过的 feeds。此处应有[简单方法](https://askubuntu.com/questions/1280743/how-to-import-my-thunderbird-settings-from-thunderbird-68-to-thunderbird-78)，

关闭所有 thunderbird，然后启动 78 时加上 `-Profilemanager`

```bash
thunderbird -Profilemanager
```

这时会要求选择 profile，只需要选择 68 对应的 profile 就好了。选好之后，再重新配置下学校邮箱的，则大功告成！

## TMUX

可以实现本地终端分屏。

参考 [linux 工具——终端分屏与vim分屏](http://blog.csdn.net/u010454729/article/details/49496381)

!!! info
    现在改用 `Terminator`, 又称 `X-terminal-emulator`。

还可以切换后台运行，在服务器上操作特别方便。

常用操作

```bash
# new a shell
tmux
# new a shell with name
tmux new -s NAME
# view all shell
tmux ls
# go back
tmux attach-session -t [NUM]
# simplify
tmux attach -t [NUM]
# more simplify
tmux a -t [NUM]
# via name
tmux a -t NAME
# complete reset: https://stackoverflow.com/questions/38295615/complete-tmux-reset
tmux kill-server
# rename: https://superuser.com/questions/428016/how-do-i-rename-a-session-in-tmux
Ctrl + B, $
```

refer to
- [How do I access tmux session after I leave it?](https://askubuntu.com/questions/824496/how-do-i-access-tmux-session-after-i-leave-it)
- [Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)
- [tmux cheatsheet](https://gist.github.com/henrik/1967800)

## Vi/Vim

### 复制

- 单行复制: 在命令模式下，将光标移动到将要复制的行处，按“yy”进行复制；
- 多行复制: 在命令模式下，
    - `nyy` + `p`
    - `:6,9 co 12`:复制第6行到第9行之间的内容到第12行后面。
    - 设置标签，光标移到起始行（结束行，粘贴行），输入 `ma` (`mb`, `mc`) `:'a, 'b co 'c`。

!!! tip
    将 `co` 改成 `m` 就变成剪切了。

### 删除

- 删除光标后的字符 `d$`
- `:.,$d`: 删除当前行到最后一行

参考 [How to Delete Lines in Vim / Vi](https://linuxize.com/post/vim-delete-line/)

### 去除 BOM

[BOM (byte-order mark, 字节顺序标记)](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83%E7%B5%84%E9%A0%86%E5%BA%8F%E8%A8%98%E8%99%9F) 是位于码点 `U+FEFF` 的统一码字符的名称。

> 在UTF-8中，虽然在 Unicode 标准上允许字节顺序标记的存在，但实际上并不一定需要。UTF-8编码过的字节顺序标记则被用来标示它是UTF-8的文件。它只用来标示一个UTF-8的文件，而不用来说明字节顺序。许多视窗程序（包含记事本）会需要添加字节顺序标记到UTF-8文件，否则将无法正确解析编码，而出现乱码。然而，在类Unix系统（大量使用文本文件，用于文件格式，用于进程间通信）中，这种做法则不被建议采用。因为它会妨碍到如解译器脚本开头的Shebang等的一些重要的码的正确处理。它亦会影响到无法识别它的编程语言。如gcc会报告源码档开头有无法识别的字符。

如果需要去除 BOM，直接 vim 打开，

```bash
:set nobomb
:wq
```

参考

- [Linux环境下如何将utf-8格式文件转变成无bom的utf-8格式文件？](https://segmentfault.com/q/1010000000256502)
- [「带 BOM 的 UTF-8」和「无 BOM 的 UTF-8」有什么区别？网页代码一般使用哪个？](https://www.zhihu.com/question/20167122)

### Ctrl+s 假死

vim并没有死掉，只是停止向终端输出而已，要想退出这种状态，只需按 `Ctrl + q` 即可恢复正常。

参考[vim按了Ctrl + s后假死的解决办法](http://blog.csdn.net/tsuliuchao/article/details/7553003)

### 执行当前脚本

```bash
:!%
```

其中 `%` expands current file name，另外

```bash
:! %:p
```

会指定绝对路径，而如果路径中有空格，则用

```bash
:! "%:p"
```

参考

- [How to execute file I'm editing in Vi(m)](https://stackoverflow.com/questions/953398/how-to-execute-file-im-editing-in-vim)
- [VIM中执行Shell命令（炫酷）](https://blog.csdn.net/bnxf00000/article/details/46618465)


### write with sudo

For example, as said in [How does the vim “write with sudo” trick work?](https://stackoverflow.com/questions/2600783/how-does-the-vim-write-with-sudo-trick-work)

```bash
:w !sudo tee %
```

and such reference gives a more detailed explanation for the trick.

### 打开另外一个文件

参考

1. [vim 打开一个文件后,如何打开另一个文件?](https://zhidao.baidu.com/question/873060894102392532.html)
2. [VI打开和编辑多个文件的命令 分屏操作 - David.Wei0810 - 博客园](https://www.cnblogs.com/david-wei0810/p/5749408.html)

### 对每行行首进行追加、替换

按住 v 或者 V 选定需要追加的行，然后再进入 `:` 模式，输入正常的 `sed` 命令，如

```bash
s/^/#/g
```

参考 [Ubuntu 下对文本文件每行行首进行追加、替换](http://blog.csdn.net/u010555688/article/details/48416765)

## VS Code

### Fail to open terminal

首先通过搜索图形界面登录，弹出

- resolving your shell environment is taking too long...
- unable to resolve your shell environment...

详见 [Resolving Shell Environment is Slow (Error, Warning)](https://code.visualstudio.com/docs/supporting/faq#_resolving-shell-environment-is-slow-error-warning)

只是按照其提示检查了 `~/.bashrc`，没有问题。

然后试着在命令行中输入 `code` 启动，此时试图打开 terminal 并没有上述信息弹出，然而 terminal 还是无法打开，在开启新 terminal 那里可以选择 log，所以当我新开一个 terminal 时，发现同时弹出下面错误消息，

```bash
[2021-03-11 10:40:29.231] [renderer1] [error] A system error occurred (EACCES: permission denied, open '/proc/1/environ'): Error: EACCES: permission denied, open '/proc/1/environ'
```

然后发现在 [terminal.integrated.inheritEnv breaks integrated terminal #76542](https://github.com/microsoft/vscode/issues/76542#issuecomment-589768136) 中提到了 enable terminal.integrated.inheritEnv 就好。

打开 setting，然后直接输入 `@modified` 快速进入更改过的设置，其中便有 inheritEnv 这一项，enable 之后重新在命令行中启动 code，此时可以打开 terminal，但是出现了以下信息

```bash
$ bind: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 18888
bind: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 18889
Could not request local forwarding.
Warning: remote port forwarding failed for listen port 30013
Warning: remote port forwarding failed for listen port 24800
```

这是为了[启动时自动登录服务器转发端口的程序](asAserver.md#boot-into-text-mode)，这时意识到 vscode 在启动时，应该会调用 `.profile`，但是我把自动连接服务器的程序写进了 `.profile`，解决办法便是复制一份 `.profile` 至 `.bash_profile`, 只在后者中保留自动登录程序，因为[启动时后者调用顺序更高](software.md/#_8)。

> .profile is for things that are not specifically related to Bash, like environment variables $PATH it should also be available anytime. .bash_profile is specifically for login shells or shells executed at login.
> source: [What is the difference between ~/.profile and ~/.bash_profile?](https://unix.stackexchange.com/questions/45684/what-is-the-difference-between-profile-and-bash-profile)

然而！似乎 vscode 启动时还是会调用 `.bash_profile`，只有在 `.bash_profile` 将自动登录程序去掉才能打开 terminal。

可能原因应该是 terminal 实际上在运行 `/bin/bash` 程序，而打开 bash 其还是按照正常打开顺序来的，如 `man bash` 中所说，

```bash
--noprofile
       Do  not read either the system-wide startup file /etc/profile or any of the personal initialization files ~/.bash_profile, ~/.bash_login, or ~/.profile.  By default,
       bash reads these files when it is invoked as a login shell (see INVOCATION below).
```

干脆换一种自启动方式，[How to run scripts on start up?](https://askubuntu.com/questions/814/how-to-run-scripts-on-start-up)

通过 `crontab -e` 设置自启动任务，

```bash
@reboot sh /home/weiya/rssh4lab.sh &
```

然而第二次重启时，并不起作用，查看　`/var/log/syslog`，原因在于网络连接在于 ssh 之后

```bash
Mar 12 09:14:23 weiya-ThinkPad-T460p CRON[1464]: (weiya) CMD (sh /home/weiya/rssh4lab.sh &)
Mar 12 09:19:37 weiya-ThinkPad-T460p NetworkManager[1342]: <info>  [1615511977.5232] device (wlp3s0): Activation: (wifi) Stage 2 of 5 (Device Configure) successful.  Connected to wireless network 'CUHK1x'.
```

接下来有两个策略，参考 [How do I start a Cron job 1 min after @reboot?](https://unix.stackexchange.com/questions/57852/how-do-i-start-a-cron-job-1-min-after-reboot)

- sleep 一定时间：这一点可以加上 `sleep 60 &&`，但是 sleep 多少并不太好把握，一般来说会在进入用户界面后网络才连接，而有时候并没有及时进入用户界面
- 判断网络好了之后在执行，上述问题回答下有人指出通过 `systemd` 来实现，其中可以指定 `After=network.target`，按照指示设置完毕，然后还是不行，原因应该是上述所说，只有进入用户界面才会连接网络，所以在系统层面的设置太早了

另外，同时也把 ssh 替换成 autossh，然后又回到之前[推荐 crontab 的问题中](https://askubuntu.com/questions/814/how-to-run-scripts-on-start-up)，采用第一种 `upstart` 方法，因为其可以在用户层面进行设置，但是设置完毕后在 syslog 中都没有相关运行命令，然后发现这种方法已经过时了。

既然 upstart 有 system 和 user 两个层面的方法，那么 systemd 应当也可以在用户层面进行设置，于是我找到了这个回答，[How to start a systemd service after user login and stop it before user logout](https://superuser.com/questions/1037466/how-to-start-a-systemd-service-after-user-login-and-stop-it-before-user-logout)，这个方法在这个回答，[How to run scripts on start up?](https://askubuntu.com/a/719157)，中也提到了 。

已经看到成功的希望了，不过还是报错了，

```bash
$ vi /var/log/syslog
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Starting ssh to lab with port forward...
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Started ssh to lab with port forward.
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Reached target Default.
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Startup finished in 70ms.
Mar 12 12:00:03 weiya-ThinkPad-T460p rssh4lab.sh[2722]: ssh: connect to host XX.XX.XX.XX port 22: Network is unreachable
Mar 12 12:00:03 weiya-ThinkPad-T460p autossh[2724]: ssh exited prematurely with status 255; autossh exiting
```

查看 `autossh` 的帮助文档发现，

```bash
AUTOSSH_GATETIME
        Specifies how long ssh must be up before we consider it a successful connection. The default is 30 seconds. Note
        that if AUTOSSH_GATETIME is set to 0, then not only is the gatetime behaviour turned off, but autossh also ignores
        the first run failure of ssh. This may be useful when running autossh at boot.
```

而 `-f` 可以达到这个效果，所以试着加上 `-f`,便成功了！

查看相关运行信息，

```bash
$ journalctl --user -u ssh4lab.service
-- Logs begin at Thu 2020-12-24 01:44:48 CST, end at Fri 2021-03-12 12:33:18 CST. --
Mar 12 12:00:03 weiya-ThinkPad-T460p autossh[2724]: starting ssh (count 1)
Mar 12 12:00:03 weiya-ThinkPad-T460p autossh[2724]: ssh child pid is 2725
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Starting ssh to lab with port forward...
Mar 12 12:00:03 weiya-ThinkPad-T460p systemd[2710]: Started ssh to lab with port forward.
Mar 12 12:00:03 weiya-ThinkPad-T460p autossh[2724]: ssh exited prematurely with status 255; autossh exiting
Mar 12 12:09:11 weiya-ThinkPad-T460p systemd[2710]: Stopped ssh to lab with port forward.
-- Reboot --
Mar 12 12:10:29 weiya-ThinkPad-T460p autossh[2649]: starting ssh (count 1)
Mar 12 12:10:29 weiya-ThinkPad-T460p autossh[2649]: ssh child pid is 2650
Mar 12 12:10:29 weiya-ThinkPad-T460p systemd[2634]: Starting ssh to lab with port forward...
Mar 12 12:10:29 weiya-ThinkPad-T460p systemd[2634]: Started ssh to lab with port forward.
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh exited with error status 255; restarting ssh
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: starting ssh (count 2)
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh child pid is 2655
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh exited with error status 255; restarting ssh
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: starting ssh (count 3)
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh child pid is 2656
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh exited with error status 255; restarting ssh
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: starting ssh (count 4)
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh child pid is 2657
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh exited with error status 255; restarting ssh
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: starting ssh (count 5)
Mar 12 12:10:30 weiya-ThinkPad-T460p autossh[2649]: ssh child pid is 2658
```

其中 `reboot` 所在行上面的为没有添加 `-f` 选项时的日志，可以发现内容与上面查看 syslog的差不多，只不过没有那么全，比如没有指出 `Network is unreachable`，而后面添加了 `-f` 选项后，在失败重试若干次后，成功了！


## WeChat in Linux

起因是今天网页端竟然登不上去，本来觉得用不了就算了吧，正好降低聊天时间，但是想到很多时候传传文件大家还是习惯用微信，所以还是准备捣鼓下 linux 版。我记得之前试过一种，但是那似乎也是基于网页版的，只是封装了一下。而今天看到了基于 wine 以及将其打包成 docker 的解决方案！

docker 了解一点，知道如果成功，以后安装卸载会很简单，于是使用 [huan/docker-wechat](https://github.com/huan/docker-wechat) 提供的 docker image，但是后来[输入时文本不可见的问题](https://github.com/huan/docker-wechat/issues/40)很恼人 ，也不知道怎么解决。

注意到作者的 docker 是在 19.10 上构建的，在想会不会与我的 18.04 不够兼容，所以准备自己修改 docker，其实都已经 fork 好了，但是由于 [wine 对 18.04 的支持有个问题](https://forum.winehq.org/viewtopic.php?f=8&t=32192)，虽说可能跟输入法也不太有关，但是还是试着装这个，后面改写 docker file 时重新 build 总是出问题，一直没解决，所以决定放弃。

于是差不多想放弃 docker 了，想直接安装 wine，弊端似乎也就是卸载会有点繁，但是如果安装成功，那就用着呗，也不用卸载了。

于是参考 [WeChat Desktop on Linux](https://ferrolho.github.io/blog/2018-12-22/wechat-desktop-on-linux)

1. [install WineHQ](https://wiki.winehq.org/Ubuntu_zhcn)

```bash
The following packages have unmet dependencies:
 gstreamer1.0-plugins-good : Breaks: gstreamer1.0-plugins-ugly (< 1.13.1) but 1.8.3-1ubuntu0.1 is to be installed
 winehq-stable : Depends: wine-stable (= 5.0.0~bionic)
E: Error, pkgProblemResolver::Resolve generated breaks, this may be caused by held packages.
```

solution

```bash
# (re)install gstreamer1.0-plugins-good and gstreamer1.0-plugins-ugly
sudo apt-get install gstreamer1.0-plugins-good
sudo apt-get install gstreamer1.0-plugins-ugly
```

```bash
Error: winehq-stable : Depends: wine-stable (= 5.0.0~bionic)
```

It is due to [FAudio for Debian 10 and Ubuntu 18.04](https://forum.winehq.org/viewtopic.php?f=8&t=32192), and

> The quickest and easiest way to satisfy the new dependency is to download and install both the i386 and amd64 libfaudio0 packages before attempting to upgrade or install a WineHQ package.

seems does not work. I need to add the repository as suggested in [Error: winehq-stable : Depends: wine-stable (= 5.0.0~bionic)](https://nixytrix.com/error-winehq-stable-depends-wine-stable-5-0-0-bionic/)

```bash
curl -sL https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/Release.key | sudo apt-key add -
sudo apt-add-repository 'deb https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/ ./'
```

but seems not due to this, and follow the instruction in [docs/WineDependencies.md](https://github.com/lutris/docs/blob/master/WineDependencies.md)

```bash
sudo apt-get install libgnutls30:i386 libldap-2.4-2:i386 libgpg-error0:i386 libxml2:i386 libasound2-plugins:i386 libsdl2-2.0-0:i386 libfreetype6:i386 libdbus-1-3:i386 libsqlite3-0:i386
```

then the problem is solved. And continue to follow the steps in [WeChat Desktop on Linux](https://ferrolho.github.io/blog/2018-12-22/wechat-desktop-on-linux)

### 窗口轮廓阴影

当从微信切换到其他软件时，会留下一个窗口轮廓阴影。再一次感叹 google 的强大，本来这个问题我都不知道怎么搜索，但只给了 “wine wechat” 和 “窗口轮廓” 这两个关键词后，就找到了两种解决方案：

- [解决Linux下微信透明窗口的问题](https://manateelazycat.github.io/linux/2019/09/29/wechat-transparent-window.html)：切换微信后，直接关掉窗口，侧边栏也不在有微信的窗口，再次启动需要点击顶部栏的图标
- [wine-wechat 窗口阴影置顶解决方案](https://www.wootec.top/2020/02/16/wine-wechat%E9%98%B4%E5%BD%B1%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)：通过 `xdotool` 去除窗口阴影。之前折腾 docker-wechat 就看到[有人提到 `xdotool`](https://www.kpromise.top/run-wechat-in-linux/)

更偏向第二种策略，后来尝试了一下也确实觉得第二种好用，当然我都有折腾过。

其实策略本身很简单，而且都可以即时运行一下脚本，就能感受效果，但是在脚本自动化那里花了挺长时间。

策略二本来推荐的是通过修改 wechat 的启动程序，但是因为我的启动程序是通过 wine 实现的，这一点似乎跟博客中不一样，我只找到一个 `WeChat.desktop` 文件，也没把握直接修改。所以我先去尝试了开机自启动

#### 开机自启动

实现方案有很多，但总想找种最简单的，有考虑 systemd service， [How do I run a script as sudo at boot time on Ubuntu 18.04 Server?](https://askubuntu.com/questions/1151080/how-do-i-run-a-script-as-sudo-at-boot-time-on-ubuntu-18-04-server)

中间也有试过 `/etc/init.d`，[Ubuntu下添加开机启动脚本](https://blog.csdn.net/hcx25909/article/details/9068497)，但是报出 warning

> warning: /etc/init.d/test missing LSB information

有点不放心。

另外还想到直接在 `.profile` 中添加运行程序，这应该算是最简单的方式，这在 [Ubuntu添加和设置开机自动启动程序的方法](https://blog.csdn.net/qq_14989227/article/details/79227283) 中有更系统的总结。

但是后来才发现，开机自启动并不适用于策略二提供的脚本，因为在判断没有微信时，程序会自动退出去。本来也想着简单改下使其能够始终运行，但是觉得这种不如原作者的更优，白白占用内存。

### 修改 desktop 文件

这个关键点在于，使其能够同时运行 wine 和 `disable-wechat-shadow.py` 脚本文件，原 `WeChat.desktop` 文件为

```bash
Exec=env WINEPREFIX="/home/weiya/.wine32" wine C:\\\\windows\\\\command\\\\start.exe /Unix /home/weiya/.wine32/dosdevices/c:/ProgramData/Microsoft/Windows/Start\\ Menu/Programs/WeChat/WeChat.lnk
```

第一种尝试便是直接加上 `& python3 /home/weiya/disable-wechat-shadow.py`，但是似乎当做了 wine 的 argument，这一点[@mango](https://askubuntu.com/a/461495)也指出

> The `Exec` key must contain a command line. A command line consists of an executable program optionally followed by one or more arguments.

第二种便是通过 `sh -c`，即

```bash
Exec=env WINEPREFIX="/home/weiya/.wine32" sh -c "wine C:\\\\windows\\\\command\\\\start.exe /Unix /home/weiya/.wine32/dosdevices/c:/ProgramData/Microsoft/Windows/Start\\ Menu/Programs/WeChat/WeChat.lnk; python3 /home/weiya/disable-wechat-shadow.py"
```

但是这个经常报出错误

> key "Exec" in group "Desktop Entry" contains a quote which is not closed

很纳闷，觉得这不应该啊，而且也[有人 wine 的解决方案](https://askubuntu.com/questions/1200116/desktop-error-there-was-an-error-launching-the-application)也是通过 `sh -c`，但是后来发现了个区别，别人引号中空格只需要一次转义，即 `\ `，比如 "Program\ Files\ (x86)"，但是我这里原先没加引号时，用两个反斜杠进行转义 `\\ `，而且 `\\` 需要用 `\\\\` 来转义，所以隐隐觉得可能是加了引号转义会有问题。所以尝试改成 `\ ` 和 `\\`，但是都没有成功。

!!! tip
    一般地，`sh -c "command 1; command 2"`可以实现在 launcher 中同时运行两个命令，参考 [How to combine two commands as a launcher?](https://askubuntu.com/questions/60379/how-to-combine-two-commands-as-a-launcher)

后来跑去研究下 `wine` 的命令，想弄清楚那一长串 argument 是什么意思，才明白这应该是句 `wine start`，而 `/Unix` 是为了用 linux 格式的路径，详见 [3.1.1 How to run Windows programs from the command line](https://wiki.winehq.org/Wine_User%27s_Guide)，所以为了避免可能的转义问题，首先可以把 `wine C:\\\\windows\\\\command\\\\start.exe` 替换成 `wine start`，而空格转义还是用 `\ `，即最终 `WeChat.desktop` 文件为

```bash
Exec=env WINEPREFIX="/home/weiya/.wine32" sh -c "wine start /Unix /home/weiya/.wine32/dosdevices/c:/ProgramData/Microsoft/Windows/Start\ Menu/Programs/WeChat/WeChat.lnk; python3 /home/weiya/disable-wechat-shadow.py"
```

这个版本终于成功了！

注意到 `env` 不要删掉，虽然在 `.bashrc` 中有设置，但是通过 desktop 启动时，并不会 source `.bashrc`，所以仍需保留这句设置，不然 `wine` 会找不到。

### window id

策略二基于的假设是，

> 微信窗口后四位所对应的不同窗口层次是固定的. 主窗口是0xXXXX000a, 那么阴影所对应的窗口就是0xXXXX0014.

而且确实好几次我阴影对应的窗口就是 `0xXXXX0014`，所以直接用了代码。但是后来发现，有时代码不起作用，这时才意识到可能 id 没对上。果然，这时候变成了 `0xXXXX0015`。不过，“不同窗口层次是固定的” 这个规律仍适用，而且我发现刚好差 8 （虽然这一点对原作者好像不适用），所以把第 25 行改成

```python
shadow = hex(int(id, 16) + 8)
```

顺带学一下 python 怎么处理十六进制，`hex()` 会把十进制数转化为十六进制，并以 `0x` 开头的字符串表示。

到这里，这个问题差不多是解决了。

#### update@20210118

这两天更新 wine 到了 6.0，然后发现窗口轮廓阴影又出现了。后来检查发现是确定 wechat 窗口的语句变化了，之前是

```python
if item.find("wechat.exe.Wine") != -1:
```

这能跟记录死机时记录的 `/var/log/syslog` 对得上，

![](wechat-window-before.png)

但是现在运行

```bash
$ wmctrl -l -G -p -x
```

发现这句变成了

```bash
0x0680000c  0 12559  870  596  1238 738  wechat.exe.wechat.exe  weiya-ThinkPad-T460p 微信
```

所以将上述去除轮廓阴影的代码改成了

```python
if item.find("wechat.exe") != -1:
```

代码详见 [disable-wechat-shadow.py](disable-wechat-shadow.py)

### cannot send images

!!! done
    当前微信 3.0.0.57 版本中，这个问题已经解决了！

Try to use the approach

```bash
sudo apt install libjpeg62:i386
```

suggested in [微信无法发送图片可以尝试一下这个方法 #32](https://github.com/wszqkzqk/deepin-wine-ubuntu/issues/32), but it does not work after `wineboot -u` and `winboot -r` and system reboot. And I even install `libjpeg8:i386` and `libjpeg9:i386`, still does not work, and then I doubt if I miss other dependencies, such as [debian-special-pkgs/deepin-wine_2.18-12_i386/DEBIAN/control](https://github.com/wszqkzqk/deepin-wine-ubuntu/commit/5300834405de1388893f2cedeb5c74f6b307a4f8#diff-4398b218af11cf74c553720d61cdff90), but I the `libjpeg-turbo8` and `libjpeg-turbo8:i386` had been installed, then I had no idea.

Notice that one comment in [微信无法发送图片可以尝试一下这个方法 #32](https://github.com/wszqkzqk/deepin-wine-ubuntu/issues/32#issuecomment-617709981)

> arch没有问题的，禁用ipv6就行了

and also the commands related to ipv6 in [如何优雅地在Ubuntu下使用QQ微信](https://zhuanlan.zhihu.com/p/91327545), then I try to disable ipv6. Here are two approaches,

- modify `/etc/sysctl.conf`
- modify GRUB

more details can be found in [How to Disable IPv6 in Ubuntu Server 18.04/16.4 LTS](https://www.configserverfirewall.com/ubuntu-linux/ubuntu-disable-ipv6/) or [在Linux下禁用IPv6的方法小结](https://www.cnblogs.com/MYSQLZOUQI/p/6232475.html)

But the first method seems not work after reboot, and need to run `sudo sysctl -p`. Then I found that when I run the ssh script to establish reverse tunnel, it reports that the address cannot be assigned, but actually it indeed works, then I realized that ssh would try to assign address for ipv4 and ipv6 simultaneously. It also reminds me [a solution](https://serverfault.com/questions/444295/ssh-tunnel-bind-cannot-assign-requested-address) found several days ago, adding `-4` for specifying ipv4.

However, this method seems also not work.

### DLL file

No clear idea about DLL file, such as `ole32.dll` suggested in [wine运行windows软件](https://jerry.red/331/wine%e8%bf%90%e8%a1%8cwindows%e8%bd%af%e4%bb%b6), this page， [Windows 7 DLL File Information - ole32.dll](https://www.win7dll.info/ole32_dll.html), might helps.

And general introduction for DLL can be found in [DLL文件到底是什么，它们是如何工作的？](https://cloud.tencent.com/developer/ask/69913)

- 类似 `.so`
- 在Windows中，文件扩展名如下所示：静态库（`.lib`）和动态库（`.dll`）。主要区别在于静态库在编译时链接到可执行文件; 而动态链接库在运行时才会被链接。
- 通常不会在计算机上看到静态库，因为静态库直接嵌入到模块（EXE或DLL）中。动态库是一个独立的文件。
- 一个DLL可以在任何时候被改变，并且只在EXE显式地加载DLL时在运行时加载。静态库在EXE中编译后无法更改。一个DLL可以单独更新而无需更新EXE本身。

## Zotero

### Tips when saving

- https://www.sciencedirect.com: press the save button on the page of the article, and no need to go to the Elsevier Enhanced Reader page by clicking "Download pdf", otherwise the saved item does not properly show the bib info and the type becomes webpage instead of journal

![](https://user-images.githubusercontent.com/13688320/114495190-3c6c9d80-9c50-11eb-9d24-dda8ee02acc7.png)

![](https://user-images.githubusercontent.com/13688320/114495254-560de500-9c50-11eb-8ab0-aa415b3608c0.png)
