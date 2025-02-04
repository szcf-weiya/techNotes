---
comments: true
---

# Application Software on Linux

According to [Wikipedia](https://en.wikipedia.org/wiki/Software#Purpose,_or_domain_of_use), computer software can be divided into

- Application Software
- System Software
    - Operating systems
    - Device drivers
    - Utilities

This section would contain comprehensive application software, and most of them have a GUI. 

<!--
I try to categorize them as follows,

- Text Editor: Atom, VS Code, Emacs, Vi/Vim
- Work: 
- Multimedia: 
-->


!!! tip
    The window of a software can be always set on top. First minimize the desired window, and the right click around the top menu, then select "Always on Top".

??? tip "make software searchable"

    If the software has `xx.destop` file, then

    ```bash
    cp xx.destop ~/.local/share/applications
    ```

    otherwise， create a `.desktop` file. More details refer to [How to pin Eclipse to the Unity launcher?](https://askubuntu.com/questions/80013/how-to-pin-eclipse-to-the-unity-launcher) and [How to add programs to the launcher (search)?](https://askubuntu.com/questions/285951/how-to-add-programs-to-the-launcher-search)

??? tip "Set Default Software"

    网页文件 `.html` 默认用百度网盘打开，之前通过 `KDE System Setting` 修改了默认软件，

    ![](https://user-images.githubusercontent.com/13688320/117541554-fb955800-b046-11eb-8577-f39fdbf406bc.png)

    但似乎并没有解决问题。

    试着参考 [Open files with other applications](https://help.ubuntu.com/stable/ubuntu-help/files-open.html.en) 的步骤进行设置

    - 右键选择 `Properties`
    - 然后选择 `Open With`
    - 选择特定软件，`Set as default`

## Atom

??? warning "Uninstalled"

    #### atom 自动更新

    [atom](https://launchpad.net/~webupd8team/+archive/ubuntu/atom/)

    ```
    sudo add-apt-repository ppa:webupd8team/atom
    sudo apt-get update
    ```

    #### proxy 设置

    Atom 的包管理器是 [apm](https://github.com/atom/apm#behind-a-firewall)，其中有介绍怎么设置 proxy，即

    ```bash
    apm config set strict-ssl false
    apm config set http-proxy http://127.0.0.1:8118
    ```

    #### spell-check

    By default, one of the core package [spell-check](https://github.com/atom/spell-check) does not check `.tex` file, although there is [another package](https://github.com/AtomLinter/linter-spell-latex) for checking the spelling in `.tex`, it does not work and not recently update, then I investigate the `spell-check` package to add the support for `.tex`. Actually, it is quite easy, just to add the scope of the `.tex` file, which can be found by `Editor: Log Cursor Scope`. So I add `text.tex.latex`, but it would be annoying to highlighter the native latex command, such as `\newcommand`, then I found that there is a `Excluded Scopes` in the config page of `spell-check`, so we only need to add the scope name of such native latex command, which again can be found by `Editor: Log Cursor Scope` if we put the cursor on the line of the commands. Finally, I add

    ```bash
    meta.preamble.latex, punctuation.definition.arguments.end.latex, support.function.general.tex, support.type.function.other.latex, storage.type.function.latex, markup.underline.link.https.hyperlink
    ```

    to the whitelist, each of which is identified by tries, such as cannot continue to add `meta.group.braces.tex` since the original text would also cannot be checked.

    #### Soft wrap

    Soft wrap is proper for `.tex` file, or `.md` file, but not necessary for the programming file. We can turn off the soft wrap globally in `Setting > Editor`, and actually we can reset it for each language, which can be toggled in `Setting > Package > language-<language name> > Soft Wrap`.

    refer to [Toggle Soft Wrap by Default?](https://discuss.atom.io/t/toggle-soft-wrap-by-default/58911/5)

    #### minimap

    装好用了一晚上，但是第二天用的时候却怎么也打不开了，尝试设置 Key binding，即便已经设置为了自动启动，所以原因并不是这个。

    后来通过 `apm` 安装低版本便成功了！

    ![](minimap.png)

    #### terminal

    之前一直在使用 `Platformio Ide Terminal v2.10.1`, 但是最近一段时间经常打不开 terminal，后来在其 repo issue 中看到类似的问题，然后有人指出这个 package 其实[不再维护](https://github.com/platformio/platformio-atom-ide-terminal/issues/543)，并且推荐了

    - terminus: https://github.com/bus-stop/terminus
    - x-terminal: https://github.com/bus-stop/x-terminal

    打不开 terminal 的原因应该与下文中提到的 VS code 类似，在替换自动启动方式之前，试过在 x-terminal 中启动程序 `/bin/bash` 添加 `--noprofile` 选项，但是报错，于是直接选择了 terminus.

    #### Toggle Symbol

    It is quite convenient to use the shortcut `Ctrl + R` to select the functions, particularly in Julia. Just want to find the equivalent behavior in VScode, the first thing is to find the official name of such a behavior. 

    I found the Keybindings table in the Setting panel, and knew that this is called "Toggle file symbols".

    Based on this hint, I found the the corresponding shortcut in VScode, that is, `Ctrl + Shift + .`, refer to [Go to next method shortcut in VSCode](https://stackoverflow.com/questions/46388358/go-to-next-method-shortcut-in-vscode)

## Baidu NetDisk

!!! info "Update 2022-11-10 21:09:28"
    下载 deb 包更新到最新版本 4.14.5，体验还行（虽然不常用）

发现百度网盘[**官方**](https://pan.baidu.com/download)出了 Linux 版，但是此时的 Linux版 V2.0.2（更新时间：2019-07-25） 在 Ubuntu 16.04 似乎运行不了——能下载安装但是无法打开运行。

在官方的 Linux 版本未发布之前，曾尝试社区开发的几种访问百度云的方案。

??? warning "Other Alternatives (Discard)"

    #### bcloud

    项目地址：[https://github.com/XuShaohua/bcloud](https://github.com/XuShaohua/bcloud)

    但是四五年没有更新了。安装试了一下，登录不了，遂放弃。

    #### PanDownload

    [https://www.baiduwp.com](https://www.baiduwp.com)

    不需要安装客户端，只需要输入网盘分享链接和提取码，便可以下载文件（而百度网盘本身下载文件需要打开客户端）。不过速度似乎不咋地

    #### bypy

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

!!! note "Highly Recommended Extensions (强烈推荐的插件)"
    - [Hypothesis](https://web.hypothes.is/): 网页标注
    - [GoFullPage - Full Page Screen Capture](https://chrome.google.com/webstore/detail/gofullpage-full-page-scre/fdpohaocaechififmbbbbbknoalclacl): 滚动截图
    - [Text Blaze](https://chrome.google.com/webstore/detail/text-blaze/idgadaccgipmpannjkmfddolnnhmeklj/related): 自定义字符串替换
        - [应用场景之一](../doc/markdown.md#collapsible-section)：在 Markdown 输入框中敲击 `/details` 即可替换成繁琐的 `<details><summary></summary></details>`（省略了必要的换行）

??? warning "not saving password"

    有段时间，不能自动输入 CUSIS 的登录信息，直接删掉 Login Data，

    ```bash
    $ pwd
    /home/weiya/.config/google-chrome/Default
    ~/.config/google-chrome/Default$ mv Login\ Data "Login-Data-backup20210410-issue16"
    ```

    详见 [:link:](https://github.com/szcf-weiya/techNotes/issues/16)

??? tip "disable running in background"

    To diagnose high memory usage, I found chrome is still running in background even if it has been closed,

    ```bash
    ~$ ps -e -o pid,cmd,%mem --sort=-%mem | grep google 
    13404 /opt/google/chrome/chrome -  1.9
    13622 /opt/google/chrome/chrome -  1.3
    632184 /opt/google/chrome/chrome -  0.7
    13448 /opt/google/chrome/chrome -  0.6
    13610 /opt/google/chrome/chrome -  0.6
    1601941 /opt/google/chrome/chrome -  0.4
    ...
    ```

    To disable it, click `Setting > Advanced > System`, and then turn off the option

    > continue running background apps when Google Chrome is closed.

    Here is a [如何看待 PC 版 Chrome 关闭后仍然可以在后台运行？ - 知乎](https://www.zhihu.com/question/21193738).

??? tip "disable reading list"

    the new version releases the `reading list`, then every time I press `star` requires to select to add to bookmarks or reading list, that make me annoyed. I found [some guys](https://www.reddit.com/r/chrome/comments/mhdn5d/how_do_i_make_it_so_when_i_hit_the_star_button_i/) have the same feeling, the solution is to enter

    ```bash
    chrome://flags/#read-later
    ```

    and then disable `reading list`.

??? tip "解决黑屏问题"

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

??? tip "Monitor website for entertainment"
    用下面脚本监控当前是否处于娱乐网站，如果是，则弹窗提醒“不要玩！”
    ```bash
    n=`wmctrl -l | grep "bilibili\|中國人線上看\|YouTube\|知乎" | wc -l`
    if [ $n -gt 0 ]; then
        notify-send -i /home/weiya/.local/share/icons/warning-icon.svg -u critical "不要玩!!!"
    fi
    ```
    将上述代码写进脚本 `check_video.sh`，并设置每10min检测一次。

    ```bash
    */10 * * * * export XDG_RUNTIME_DIR=/run/user/$(id -u); export DISPLAY=:1; sh /home/weiya/github/techNotes/docs/Linux/check_video.sh
    ```
    ![Screenshot from 2023-01-31 15-51-56](https://user-images.githubusercontent.com/13688320/215880662-d91e6775-d21a-4d50-9216-5b79b9abcef2.png)

## DroidCam: Phone as Webcam

Homepage: [DroidCam](https://www.dev47apps.com/)

First of all, install Linux client following the [official instruction](https://www.dev47apps.com/droidcam/linux/)

??? note "Mix2s"

    - install the Android app on Mix2s
    - 因为无法设置在一个局域网中，所以测试 USB 连接。根据[连接指南](https://www.dev47apps.com/droidcam/connect/)，需要打开 USB debugging，然而似乎仍然无法成功。
    - 根据错误提示运行 `adb devices`，并没有显示任何安卓设备的连接。另外 `lsusb` 并没有手机的记录，而且插上前后 `lsusb` 项目个数不变。
    - 可能电脑端缺少驱动，试图寻找 USB driver，如[www.xiaomidriversdownload.com](https://www.xiaomidriversdownload.com/xiaomi-mi-mix-2s-adb-driver/)但是只找到 for windows 的版本（后来证明并不需要，只是 USB 线的原因）。

??? note "Mi4c"

    同 Mix2s，不过换了长的那根数据线后，`lsusb` 多了条记录

    ```bash
    $ lsusb
    Bus 001 Device 033: ID 2717:ff68
    ```

    不像其它记录那样有具体的名字，找到同样的问题，[adb devices not working for redmi note 3 on ubuntu](https://stackoverflow.com/questions/40951179/adb-devices-not-working-for-redmi-note-3-on-ubuntu)。经查，该文件位于 `/lib/udev/rules.d`，下载仓库中最新的 [51-android.rules](https://github.com/M0Rf30/android-udev-rules/blob/master/51-android.rules)

    ```bash
    $ cat 51-android.rules | grep ff68
    $ sudo cp 51-android.rules 51-android.rules.old
    $ sudo cp ~/Downloads/51-android.rules .
    ```

    但是 `lsusb` 并没有立即生效，又不想重启，于是试了 [How to reload udev rules without reboot?](https://unix.stackexchange.com/questions/39370/how-to-reload-udev-rules-without-reboot)，

    ```bash
    $ udevadm control --reload-rules
    ```

    以及

    ```bash
    $ pkill -HUP udevd
    ```

    但是仍没有显示名字。

??? note "iPad"

    - firstly, install the APP
    - 因为插上 iPad 后，自动跳出是否信任本设备，而且在 `lsusb` 中找到记录 `Bus 001 Device 018: ID 05ac:12ab Apple, Inc. iPad 4/Mini1`。
    - 然后在电脑端开启连接，这样就能使用ipad的摄像头了。在zoom中，开启摄像头那里有切换至 Droidcam 的选项。

## Geeqie

Install with

```bash
sudo apt install geeqie
```

which can show the pixel info, and is first used when developing the script for the [perspective transformation](#perspective-transformation).

Refer to [Which image viewer is able to show coordinates?](https://askubuntu.com/questions/298877/which-image-viewer-is-able-to-show-coordinates)

## Google Drive

refer to [Ubuntu 16.04 set up with google online account but no drive folder in nautilus](https://askubuntu.com/questions/838956/ubuntu-16-04-set-up-with-google-online-account-but-no-drive-folder-in-nautilus)

Note that you should run

```bash
gnome-control-center online-accounts
```

in the command line, not to open the GUI.

## ImageMagick

??? tip "Perspective Transformation"

    I usually take many photos when listening to the seminars, it is desirable to extract only the slides and discard the nuisance background. Direct cropping is not enough since the photos are not parallel to the screen. 

    The solution is called perspective transformation, which can be done via [`-distort method`](http://www.imagemagick.org/script/command-line-options.php#distort), the usage is 

    ```bash
    $ magick input.jpg -distort perspective 'U1,V1,X1,Y1 U2,V2,X2,Y2 U3,V3,X3,Y3 ... Un,Vn,Xn,Yn' output.jpg
    ```

    where `U,V` on the source image is mapped to `X,Y` on the destination image.

    The interesting area is the mapped sub region, and so we need to further crop them out, which can be done with [`-crop geometry`](http://www.imagemagick.org/script/command-line-options.php#crop), where the geometry is defined with `WxH+x+y`, which means the region of size `WxH` located at the xy-coordinates `(x, y)`, see more details in [Anatomy of the Command-line](http://www.imagemagick.org/script/command-line-processing.php#geometry)

    Refer to the [source code](https://github.com/szcf-weiya/techNotes/blob/master/src/persp.py) for more details.

    !!! tip
        The original record on the development is [here](https://github.com/szcf-weiya/en/issues/191).

    - Demo One: processing a single file

    ![](https://user-images.githubusercontent.com/13688320/125592307-fbeb3be2-64e6-414f-8748-5045fca2a0e6.gif)

    - Demo Two: processing multiple files in a folder

    ![](https://user-images.githubusercontent.com/13688320/125592319-93aef031-4492-4812-934d-4ed3fcbc792a.gif)

    Some references:

    - [Displaying the coordinates of the points clicked on the image using Python-OpenCV](https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/)
    - [4 Point OpenCV getPerspective Transform Example](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

    !!! todo
        Wrap it into a GUI program.
        Possible references:

        - [OpenCV with Tkinter](https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/)
        - [Julia Package for UI development](https://discourse.julialang.org/t/julia-package-for-ui-development/39469/3)
        - [How to make a GUI in Julia? - Stack Overflow](https://stackoverflow.com/questions/35328468/how-to-make-a-gui-in-julia)

??? tip "Convert .heic to .jpg"
    ```bash
    $ sudo apt install libheif-examples
    $ heif-convert A.heic A.jpg
    ```
    In batch mode, run
    ```bash
    $ find . -name '*.heic' -exec heif-convert {} {}.jpg \;
    ```

??? fail "Add HEIC support in ImageMagick"
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

## Image Viewer 

The default software for viewing images. Start it from command line by `eog`.

- set transparent background for `png` image

default | desired 
--- | ---
![image](https://user-images.githubusercontent.com/13688320/173728951-39ad9ae1-f792-4675-8bae-e9f195723ad9.png) | ![image](https://user-images.githubusercontent.com/13688320/173728964-2a581414-278c-4d1a-835a-97005d7775aa.png)

In a word, change the default setting "as check pattern" to a white custom color.

## Input Methods for Chinese

??? note "fcitx-sougou"

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

??? note "fcitx-googlepinyin"

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

??? note "fcitx-baidu"

    既然添加个输入法这么简单，那索性再试试其它的，百度输入法可以在其官网中下载的到 `.deb` 文件，然后安装并重启输入法。

    正如上述知乎回答提到的，它乱码了！

??? note "fcitx-rime"

    这主要是繁体中文，不过似乎应该也能切换简体。本身这是基于 ibus 的，不过 [fcitx 团队](https://github.com/fcitx)有在维护 fcitx 的版本，

    ```bash
    $ sudo apt install fcitx-rime
    ```

    因为想同时比较其与谷歌拼音的体验，所以目前同时保留了这两个输入法，可以通过 `SHIFT+CTRL` 快速切换输入法。

    RIME 默认是繁体的，可以通过 `` CTRL+` `` 来切换简繁体，另外也有全半角等设置。

    !!! note
        除了这些在 fcitx4 上的方案，也许过段时间会尝试[更新的输入法框架 fcitx5](https://www.zhihu.com/question/333951476)

    虽然谷歌拼音和fcitx-rime都表现得不错，但是默认的 UI 实在有点丑，看到 [kimpanel](https://fcitx-im.org/wiki/Kimpanel) 会比较好看，想试一试，采用 gnome-shell 安装，但是竟然 [no popup window](https://github.com/wengxt/gnome-shell-extension-kimpanel/issues/53)，虽放弃。

!!! tip "ibus-rime (Currently using)"

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

    Tips:

    - 输入外国姓名中的点：中文输入法状态下 `Shift + \`
    - 自定义快捷键，取消 `Ctrl + `` 的快捷键，在 `~/.config/ibus/rime` 新建文件 `default.custom.yaml`，然后写入

    ```bash
    patch:
    "switcher/hotkeys":  # 這個列表裏每項定義一個快捷鍵，使哪個都中
        - F4
    ```

    其实也就是删去 `Control+grave`，详见 [一例、定製喚出方案選單的快捷鍵](https://github.com/rime/home/wiki/CustomizationGuide#%E4%B8%80%E4%BE%8B%E5%AE%9A%E8%A3%BD%E5%96%9A%E5%87%BA%E6%96%B9%E6%A1%88%E9%81%B8%E5%96%AE%E7%9A%84%E5%BF%AB%E6%8D%B7%E9%8D%B5)。修改完成后需要点击右上角输入法菜单中的 “部署”。

    - 输入特殊字符（比如带声调的拼音符号）：通过 F4 切换至 “朙月拼音”，然后按 “/py” 便出现带声调的拼音字符。更一般地，对于其它特殊符号，根据配置文件 `~/.config/ibus/rime/symbols.yaml` 的定义进行输入。

## Kazam

Ubuntu 下 kazam 录屏 没声音解决方案

[http://www.cnblogs.com/xn--gzr/p/6195317.html](http://www.cnblogs.com/xn--gzr/p/6195317.html)

??? warning "video format: cannot open in Windows"

    solution

    ```bash
    ffmpeg -i in.mp4 -pix_fmt yuv420p -c:a copy -movflags +faststart out.mp4
    ```

    refer to [convert KAZAM video file to a file, playable in windows media player](https://video.stackexchange.com/questions/20162/convert-kazam-video-file-to-a-file-playable-in-windows-media-player)

## ksnip

!!! info
    Post: 2021-10-02 10:54:23

Since `shutter` has stopped developing, and it seems not friendly on Ubuntu 20.04, find the alternative, [Ksnip](https://github.com/ksnip/ksnip)

Install it via `snap`,

```bash
~$ sudo apt-get install ksnip
[sudo] password for weiya: 
Reading package lists... Done
Building dependency tree       
Reading state information... Done

No apt package "ksnip", but there is a snap with that name.
Try "snap install ksnip"

E: Unable to locate package ksnip
~$ snap install ksnip
ksnip 1.9.1 from Damir Porobic (dporobic) installed
```

But note that it cannot access the external disk drive as in Okular. Hopefully, use the following line

```bash
snap connect ksnip:removable-media
```

can enable the access to external disk.

Generally, it requires the software shipped with `removable-media` plug, as mentioned in [How to get access to USB-storage from an application installed as Snap? - Ask Ubuntu](https://askubuntu.com/questions/1034030/how-to-get-access-to-usb-storage-from-an-application-installed-as-snap)

## nautilus

It is a file manager for GNOME.

??? tip "Template: new document in right click menu"

    Surprisingly, no "new document" in the right-click menu on Ubuntu 18.04, then I found the [post](https://itsfoss.com/add-new-document-option/) which also mentioned this inconvenience, and it gives a wonderful solution. Use `Templates`!! Never use this folder before!

    Just create an empty file, say `README.md`, then there will be a `New Document` in the right-click menu.

    Also check the [official documentation](https://help.ubuntu.com/stable/ubuntu-help/files-templates.html.en) on `Templates`

    > A file template can be a document of any type with the formatting or content you would like to reuse. 

??? note "Google Drive"

    短暂尝试过，但已弃用。

    可以添加 Google Drive 的帐号，从而直接在文件管理系统中访问 Google Drive 的内容。另见 [:fontawesome-brands-stack-exchange:](https://askubuntu.com/questions/838956/ubuntu-16-04-set-up-with-google-online-account-but-no-drive-folder-in-nautilus)

## Octave

??? note "Octave"

    参考[Octave for Debian systems](http://wiki.octave.org/Octave_for_Debian_systems)

    另外帮助文档见[GNU Octave](https://www.gnu.org/software/octave/doc/interpreter/)

## Okular

??? note "Installation via snap vs apt"

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

??? tip "latex in annotation"

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

??? tip "自定义签名 customized signature"

    可以通过 `stamp` 功能自定义签名，首先准备好签名图片，然后保存到某个文件夹，比如 `~/.kde/share/icons/signature.png`，然后进入 stamp 的配置界面，下拉框中直接输入签名图片所在的路径。参考 [How to add a Signature stamp to Okular](https://askubuntu.com/questions/1132658/how-to-add-a-signature-stamp-to-okular)

    但是并不能存为 pdf，或者被其他软件看到，用 Acrobat 打开会有个打叉的部分，但是看不到签名，[已经被标记为 bug，但似乎还未解决](https://bugs.launchpad.net/ubuntu/+source/okular/+bug/1859632)。

??? tip "set background color"

    set background color for visible screenshots.

    refer to [Is there a pdf reader allowing me to change background color of (arXiv) pdfs?](https://askubuntu.com/questions/472540/is-there-a-pdf-reader-allowing-me-to-change-background-color-of-arxiv-pdfs)

??? tip "remove duplicate icons"

    When opening multiple pdf files, it results duplicate icons, such as 

    ![image](https://user-images.githubusercontent.com/13688320/144483818-ea4b5393-82ea-4f0a-bd30-30e8213cf2ce.png)

    The solution is 

    - copy the desktop file

    ```bash
    mv /usr/share/applications/okularApplication_pdf.desktop .local/share/applications/
    ```

    - add the following line to the end

    ```bash
    StartupWMClass=okular
    ```

    then close all pdf files opened by okular, and re-open them, then they will be grouped into a single icon.

    Refer to 

    - [Okular instances does not group under single icon in desktop dock on Ubuntu 17.10](https://askubuntu.com/questions/995693/okular-instances-does-not-group-under-single-icon-in-desktop-dock-on-ubuntu-17-1)

    or for other software,

    - [Duplicate application icons in Ubuntu dock upon launch](https://askubuntu.com/questions/975178/duplicate-application-icons-in-ubuntu-dock-upon-launch/975230#975230)

    A by-product tip learned from the above question,

    - use `Alt + backstick` can switch the same application.

## OneDrive

In fact, the following client on Ubuntu is also in the command-line form. But usually, we refer to OneDrive as the whole of the client and the host, which is visited via a browser. 

??? note "first try"

    [xybu/onedrive-d-old](https://github.com/xybu/onedrive-d-old), but doesn't support exchange account.

??? note "second try"

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

??? note "fuseblk"

    发现使用 onedrive 同步文件时，有时候并不能够同步。猜测可能是因为文件太小，比如文件夹 `test` 中仅有 `test.md` 文件（仅70B），而此时查看 `test` 大小，竟然为 0 B，因为根据常识，一般文件夹都是 4.0k，或者有时 8.0k 等等，具体原因参考 [Why does every directory have a size 4096 bytes (4 K)?](https://askubuntu.com/questions/186813/why-does-every-directory-have-a-size-4096-bytes-4-k)

    但我现在问题是文件夹竟然是 0B，猜测这是无法同步的原因。

    后来在上述问题的回答的评论中 @Ruslan 提到

    > @phyloflash some filesystems (e.g. NTFS) store small files in the file entries themselves (for NTFS it's in the MFT entry). This way their contents occupy zero allocation blocks, and internal fragmentation is reduced. – Ruslan Nov 2 at 9:03

    猜测这是文件系统的原因，因为此时文件夹刚好位于移动硬盘中，所以可能刚好发生了所谓的 “internal fragmentation is reduced”。

    于是准备查看移动硬盘的 file system 来验证我的想法，这可以通过 `df -Th` 实现，具体参考 [7 Ways to Determine the File System Type in Linux (Ext2, Ext3 or Ext4)](https://www.tecmint.com/find-linux-filesystem-type/)

    然后竟然发现并不是期望中的 NTFS，而是 fuseblk，[東海陳光劍的博客](http://blog.sina.com.cn/s/blog_7d553bb501012z3l.html)中解释道

    > fuse是一个用户空间实现的文件系统。内核不认识。fuseblk应该就是使用fuse的block设备吧，系统中临时的非超级用户的设备挂载好像用的就是这个。

    最后发现，onedrive 无法同步的原因可能并不是因为 0 byte 的文件夹，而是因为下面的命名规范，虽然不是需要同步的文件，而是之前很久的文件，但可能onedrive就在之前这个不规范命名的文件上崩溃了。

??? tip "windows 命名规范"

    在使用 [skilion/onedrive](https://github.com/skilion/onedrive) 同步时，一直会出现碰到某个文件崩溃。查了一下才知道是需要遵循 [Windows 命名规范](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file?redirectedfrom=MSDN)，其中有两条很重要

    - Do not assume case sensitivity. For example, consider the names OSCAR, Oscar, and oscar to be the same, even though some file systems (such as a POSIX-compliant file system) may consider them as different. Note that NTFS supports POSIX semantics for case sensitivity but this is not the default behavior.
    - The following reserved characters:
    - < (less than)
    - `>` (greater than)
    - : (colon)
    - " (double quote)
    - / (forward slash)
    - \ (backslash)
    - | (vertical bar or pipe)
    - ? (question mark)
    - `*` (asterisk)

!!! tip "Change to [abraunegg/onedrive](https://github.com/abraunegg/onedrive) (Currently using)"

    I found that it will automatically run after startup, actually with [skilion/onedrive](https://github.com/skilion/onedrive), sometimes it also starts automatically. Then I tried

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

!!! tip "sync shared folder"

    refer to [How to configure OneDrive Business Shared Folder Sync](https://github.com/abraunegg/onedrive/blob/master/docs/BusinessSharedFolders.md) for full instruction.

    ```bash
    # list remote shared folder
    $ onedrive --list-shared-folders
    # configure folder to share
    $ vi ~/.config/onedrive/business_shared_folders
    # perform sync (--resync is needed when the config file has been updated)
    $ onedrive --synchronize --sync-shared-folders [--resync]
    ```

!!! tip "sync selected shared folder on Windows"
    
    Since there is unknown error for syncing shared folder on Ubuntu, I tried to download the folder manually. However, there is a limit on number of files when downloading, it is 1000. 
    
    Refer to <https://www.eduhk.hk/ocio/content/faq-how-sync-shared-me-onedrive-folders-your-local-computer>
    
    Since I dont want to sync other files, so pause the sync of other files.
    
    - in the browser, go to the folder that needs to be synced
    - click sync
    - since there is no response, click to install the latest Onedrive 
    
    
!!! tip "switch from Windows to Ubuntu: unsupported reparse point"
    
    Enable to download every files to disk in Windows. Refer to [:link:](https://askubuntu.com/questions/1345356/files-from-windows-with-unsupported-reparse-point-on-ubuntu), [:link:](https://github.com/abraunegg/onedrive/blob/master/docs/advanced-usage.md#configuring-the-client-for-use-in-dual-boot-windows--linux-situations)

## OpenShot

!!! info
    Post on: 2022-07-01 23:48:04

Homepage: <https://www.openshot.org/en/user-guide/>

Install with the official instruction:

```bash
sudo add-apt-repository ppa:openshot.developers/ppa
sudo apt update
sudo apt install openshot-qt python3-openshot
```

## Peek

[homepage](https://github.com/phw/peek), easy to use, can convert to gif.

## Planner

follow instruction in <https://flathub.org/apps/details/com.github.alainm23.planner>

first of all, configure flatpak

```bash
sudo apt install flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
```

then install planner via

```bash
flatpak install flathub com.github.alainm23.planner
```

run it by

```bash
flatpak run com.github.alainm23.planner
```

## Rhythmbox

!!! tip
    - 2022-12-07 15:25:36：不用手动 import，只要将歌曲都放在指定的目录，并勾选自动添加新歌曲。同时发现一个小 bug，如果重新用同一文件夹的不同路径名（ln -s），则会重复添加。

??? tip "修改音乐文件的 Properties"

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

!!! tip
    `exiftool music.mp3` 可以查看 meta info，而且可以修改。

??? tip "修改 .wav 文件的 Properties"

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

    #### wav -> mp3

    另一个方法便是直接将 wav 转换成 mp3，注意到如果直接用 `ls` + `xargs` 

    ```bash
    # keep old extension
    ls -1 | grep wav | head -1 | xargs -I {} ffmpeg -i {} {}.mp3
    ```

    则文件名后缀不是很好直接换掉，不能直接对 `{}` 进行操作，

    ```bash
    find . -name "*.wav" -exec sh -c 'ffmpeg -i "$1" "${1%.*}.mp3"' sh {} \;
    ```

    参考 [:link:](https://superuser.com/questions/1426601/how-to-remove-extension-from-pathname-passed-as-in-find-exec), 其实有点相当于把 `{}` 在传进去然后就可以用 `$1` 来表示。

    但是重命名完之后，Rhythmbox 左侧出现一个 Missing Files 的文件夹，里面即删掉的 `.wav` 文件路径。退出然后删掉 `rhythmdb.xml` 再打开即可，参考 [:link:](https://www.chalk-ridge.com/is-rhythmbox-missing-music-files-heres-a-simple-fix/)

    ```bash
    ~/.local/share/rhythmbox$ mv rhythmdb.xml rhythmdb.xml.old
    ```

    所以看上去路径是直接写入 `.xml` 文件中，所以即便两个不同的路径指向同一个文件夹，也会被视为不同文件夹，于是便能解释为什么相同音乐文件被重复导入。

## Synergy

??? tip "hotkey for locking client's screen from host"

    With Host@t460p(Ubuntu 18.04) and Client@STAPC(Win 10), the hotkey `Win + L` for locking the screen does not work on the client, but the hotkey would work if I use its keyboard. 

    An ideal solution would be simultaneously to lock the key via the same hotkey, as someone discussed in [Synergy: Is there a way to push Win+L to all screens, not just the server?](https://superuser.com/questions/267058/synergy-is-there-a-way-to-push-winl-to-all-screens-not-just-the-server)

    First of all, I become to use the custom behavior by defining some keystrokes via

    `Configure Server` -> `Hotkeys` -> `New Hotkey` -> `New its associated Actions`

    A complete instruction can be found in [Add a keyboard shortcut to change to different screens](https://symless.com/help-articles/add-a-hotkey-or-keyboard-shortcut-to-change-to-different-screens)

    Since `Win+L` fails on the client, try to define a new shortcut. Following the 4th method in [How to Quickly Lock Screen In Windows 10](https://www.techbout.com/lock-screen-in-windows-10-45432/),

    `Right Click` -> `New Shortcut` -> `C:\Windows\System32\rundll32.exe user32.dll,LockWorkStation`

    Then we can lock the screen after clicking this shortcut, but currently no associated hotkey. Right click to edit the properties, there is a shortcut key, which by default assumes the key starts with `Ctrl + Alt`, so just press the remaining key, say `L`. However, `Ctrl + Alt + L` does not work. 

    Instead, `Ctrl + Alt + 2` works.

    Then back to the host configuration. Bind the follow two keys,

    ```bash
    keystroke(Control+Alt+l) = keystroke(Alt+Control+2,stapc220)
    ```

    then I can lock the screen of client via `Ctrl+Alt+l`. 

    Since the above binding can specify the machine, I am thinking it might work if I bind `Ctrl+Alt+l` to `Meta+l` on the host. However, it failed, and it causes the screen of client cannot be locked.

    Anyway, the current solution seems already convinent.

## Terminator

!!! tip "Shortcuts"

    the shortcut list can be found in `Right Click >  Preferences > Keybindings`, several more common

    - resize: `Shift+Ctrl+Left/Right/Down/Up`

??? warning "Ctrl + Shift + E not work"

    - `Ctrl + Shift + E` not work in Ubuntu 20.04 + ibus. The reason is that ibus has define such shortcut for emoji annotation. So just type `ibus-setup`, then switch to Emoji tab and delete the shortcut. [:link:](https://jh-byun.github.io/study/terminator-split-vertically/)

    ![image](https://user-images.githubusercontent.com/13688320/192050782-33e0676a-d37f-47d6-a138-5b7cce58e4fa.png)


??? tip "hostname 的颜色"

    - hostname 的颜色, 去掉 `.bashrc` 中

    ```bash
    ##force_color_prompt=yes
    ```

    的注释

??? tip "hide hostname"

    - hide hostname, `weiya@weiya-ThinkPad-T460p:`

    edit the following line in the `.bashrc` as follows

    ```bash
    if [ "$color_prompt" = yes ]; then
        #PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
        PS1='\[\033[01;34m\]\w\[\033[00m\]\$ '
    else
        PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
    fi
    ```

    before hide and after hide 

    ![image](https://user-images.githubusercontent.com/13688320/123048663-37d31b00-d431-11eb-8ebf-afb97f758191.png)

??? tip "背景色"

    - 各种颜色，如背景色，直接右键设置，右键设置完成之后便有了一个配置文件，`~/.config/terminator/config`.


## Thunderbird

- 添加学校邮箱时，必须采用学号形式的邮箱，不要用 alias 形式的，alias 验证会出问题。

??? note "Esther 的 To 为啥是她自己？Bcc？"
    Post: 2022-04-02 18:12:48

    最近小伙伴们在讨论为什么 Esther 群发的邮件的 `To` 是她自己的邮箱。通过 `More > View Source` 查看邮件源码时发现，有个 `Delivered-To` 字段，恰好是 grad 的邮箱。于是猜想怎么实现这种效果，bcc？

    于是简单做了个实验，从 A 邮箱发给 B 并密送给 C

    ```bash
    From: A
    TO: B
    Bcc: C
    ```

    在 B 端查看源码发现只有 `TO: B`，而在 C 端既有 `Delivered-To: C`，也有 `TO: B`。

    所以 Esther 一种可能的原因是在 bcc 中输入 grad 的邮箱，并发送给自己。但感觉这样很奇怪，可能有其它自动设置，具体不得而知。关于 `Delivered-To` 的讨论可另见 [:link:](https://serverfault.com/questions/796913/how-can-the-to-and-delivered-to-fields-in-an-email-i-received-be-different)

??? tip "Deactive Account without Deleting"

    Since the visiting Harvard email account has expired, it will always pop up the log-in window. But I do not want to delete the account, and just want to avoid the automatically log-in, so I try to set to never check new message as follows,

    ![image](https://user-images.githubusercontent.com/13688320/151953746-2250b616-a3b7-418b-bfbc-de535224d562.png)

    refer to [deactivate email account without deleting | Thunderbird Support Forum](https://support.mozilla.org/en-US/questions/1323864)

!!! tip "Special Gmail"

    与其它邮箱帐号不同的是，添加 Gmail 后只有 Inbox 和 Trash，而没有 Sent, Drafts 等，不过有个 `[Gmail]` 文件夹，里面的子文件便有发件箱等等。这种特殊目录结构是因为对 Gmail 的不同处理方式，详见 [Special Gmail](https://support.mozilla.org/zh-CN/kb/thunderbird-gmail)

!!! tip "Proxy for Gmail in Thunderbird"

    Setting a proxy for the thunderbird is quite straigtforward, but not all mail accounts need the proxy, only gmail in my case. I am considering if it is possible to set up a proxy for gmail separately. Then I found that setting proxy by PAC file might work inspired by [Gmail imap/smtp domains to connect via proxy](https://support.google.com/mail/forum/AAAAK7un8RUCGQj5uPgJoo), since PAC file can customize the visited url.

    Then I need to learn [how to write a PAC file](https://findproxyforurl.com/example-pac-file/), although later I directly export the rules written in SwitchyOmega to a PAC file.

    Once PAC is done, I need to write its location url, seems impossible to directly write a local path. One easy way is to open port 80 to access my laptop, which maybe need apache or nginx, but both of them are overqualified. A simple way is

    ```bash
    sudo python -m SimpleHTTPServer 80
    ```

    found in[Open port 80 on Ubuntu server](https://askubuntu.com/questions/646293/open-port-80-on-ubuntu-server)

!!! note "Upgrade 68 to 78"

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

!!! tip "Train adaptive junk filter"

    more instructions: [Thunderbird and Junk / Spam Messages](https://support.mozilla.org/en-US/kb/thunderbird-and-junk-spam-messages)

## ToDesk

Homepage: <https://www.todesk.com/download_detail.html>

Currently, the Linux version is in Beta.

## Transmission 

- [Configuring Transmission for faster download - Ask Ubuntu](https://askubuntu.com/questions/110899/configuring-transmission-for-faster-download)
- [Transmission says port is closed but seeding is happening - Ask Ubuntu](https://askubuntu.com/questions/405487/transmission-says-port-is-closed-but-seeding-is-happening)

## VS Code

!!! tip "Shortcuts"
    [Official Shortcut Table](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf)

    Other shortcuts:

    - switch to next functions: `Ctrl + Shift + .`, refer to [Go to next method shortcut in VSCode](https://stackoverflow.com/questions/46388358/go-to-next-method-shortcut-in-vscode)
    - switch terminals, `Ctrl+Up/Down`, refer to [How to switch between terminals in Visual Studio Code?](https://stackoverflow.com/a/67412583/)

!!! tip "Edit Multiple Line Simultaneously"

    Press `Shift + Alt + Down/Up` to insert cursors below or up, after inserting ` >`, then press `Esc` to exit. [:link:](https://code.visualstudio.com/docs/editor/codebasics)

    ![Peek 2022-03-30 20-28](https://user-images.githubusercontent.com/13688320/160834904-5110b5e5-a58d-4586-a9a3-0e0d835abe52.gif)

!!! note "Fail to open terminal"

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

??? warning "Do not auto-programs in .profile"

    继续上个问题。

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

## w3m: Command Line Browser

Visit the website from command line without the disturbing of images,

![Peek 2021-07-15 15-38](https://user-images.githubusercontent.com/13688320/125749970-2e5f1c7e-9769-4e0d-b173-cdb154cd8d0c.gif)

Other candidates but seems not fast and smoothly,

- lynx: error in open the same website in `https`, throws "Alert!: Unable to make secure connection to remote host."
- links: cannot display Chinese characters.

Some references:

- [3个常用基于Linux系统命令行WEB网站浏览工具（w3m/Links/Lynx） | 老左笔记](https://www.laozuo.org/8178.html)
- [w3m纯文字阅读体验 - 知乎](https://zhuanlan.zhihu.com/p/351569550)

## WeChat in Linux

??? note "解码 .dat 图片"
    微信图片默认以 dat 格式存储，不能直接以图片形式打开。不过根据[网友分享](https://www.zhihu.com/question/393121310/answer/1606381900)，实际上只是用了某个 magic code 对原图片进行异或处理。虽然不同帐号不同机器的 magic code 不一样，但是我们已知常见图片存储格式 jpeg 的 header 为 `FF D8`，而用 `xxd` 查看图片的 16 进制格式发现大多数 header 为 `14 33`，于是便可得到 magic code
    ```julia
    julia> xor(0xffd8, 0x1433)
    0xebeb
    julia> xor(0xebeb, 0x1433)
    0xffd8
    ```
    所以我的 magic code 即为 `0xeb`

    其它图片格式（如 png，它的 header 为 `8950`）也共享同一个 magic code，使用 `xxd` 得到的 16 进制格式的 header 为 `62bb`

    ```julia
    julia> xor(0x62bb, 0xebeb)
    0x8950
    ```

    因为图片主要是 jpeg 格式，所以为了简单直接另存为 `.jpg`，当然可以很直接地根据 header 判断照片格式，

    ```julia
    $ ls -1 ~/WX/Image/2023-05/ | xargs -I {} ~/github/techNotes/decode_wechat_image ~/WX/Image/2023-05/{} ~/WXBP/Image/2023-05/{}.jpg
    ```

    而且即便格式错了（png 的存储为 jpg），Geeqie 仍可正常打开图片，虽然系统默认的 Image Viewer 会报错

    > Error interpreting JPEG image file (Not a JPEG file: starts with 0x89 0x50)

??? note "微信被其它人登录？"

    Post: 2022-11-05 22:49:27

    刷到这篇[讨论](https://www.zhihu.com/question/564406006/answer/2744703293)，于是跑过去看看自己微信上的登录设备记录，“Settings > Account Security > Login Devices”，然后竟然发现里面有条“Windows 7” 的记录。

    ![](https://user-images.githubusercontent.com/13688320/200734603-e0d41f2e-742b-4ff3-a75c-41c76aa58d0b.jpg)

    但目前只登录了自己的手机 Mix2s，以及第三条的笔记本 T460P。后来发现 T460P 登录的微信因为移动硬盘写入原因无响应，最后只能 kill 掉。此时一个猜测便是这个 Windows 7 的记录很可能是因为机器挂了，在获取机器型号的时候，因为未知原因，只能得到 Win7，而非 T460P.

    在图中 win7 的时间戳 "2022-11-04 17:35:26" 搜索系统日志，发现移动硬盘的 IO error 刚好在那个时间点附近，

    ```bash
    $ journalctl --since "2022-11-04 17:35" --until "2022-11-04 17:36" 
    Nov 04 17:35:23 weiya-ThinkPad-T460p ntfs-3g[2012449]: ntfs_attr_pread_i: ntfs_pread failed: Input/output error
    Nov 04 17:35:23 weiya-ThinkPad-T460p ntfs-3g[2012449]: ntfs_attr_pread error reading '/0WeChat/WeChat Files/wxid_***************/Msg/MicroMsg.db-wal' at offset 974848: 4096 <> -1: Input/output error
    Nov 04 17:35:23 weiya-ThinkPad-T460p kernel: Buffer I/O error on dev sde1, logical block 89552839, async page read
    ```

    再验证下 wine 里面是不是 Windows 7，首先运行 `winecfg` 跳出来的图形界面就有显示 Windows 7，也可以从配置文件中得到确认。

    ```bash
    ~/.wine32$ cat system.reg | grep -n "Microsoft Windows"
    995:@="Microsoft Windows Installer Message RPC"
    1028:@="Microsoft Windows Installer"
    31532:@="Microsoft Windows Installer"
    31540:@="Microsoft Windows Installer Message RPC"
    39508:"ProductName"="Microsoft Windows 7"
    ```

??? note "Use Wine WeChat on XPS with 22.04"
    1. install the stable wineHQ and setup
    ```
    export WINEARCH=win32
    export WINEPREFIX=~/.wine32
    winecfg
    ```
    2. install Chinse font for wine
    ```bash
    wget  https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks
    chmod +x winetricks
    sudo apt install cabextract
    ./winetricks corefonts gdiplus riched20 riched30 wenquanyi
    ./winetricks regedit # import registry file from https://gist.githubusercontent.com/swordfeng/c3fd6b6fcf6dc7d7fa8a/raw/0ad845f98f5a97e7173ff40b5e57b3a163e92465/chn_fonts.reg
    ```
    3. install WeChat_32bit from official site

    refer to [:link:](https://web.archive.org/web/20230603185929/https://ferrolho.github.io/blog/2018-12-22/wechat-desktop-on-linux)

??? note "Start to Use Wine Wechat (Post: ~ 2020.05.12)" 
    
    起因是今天网页端竟然登不上去，本来觉得用不了就算了吧，正好降低聊天时间，但是想到很多时候传传文件大家还是习惯用微信，所以还是准备捣鼓下 linux 版。我记得之前试过一种，但是那似乎也是基于网页版的，只是封装了一下。而今天看到了基于 wine 以及将其打包成 docker 的解决方案！

    docker 了解一点，知道如果成功，以后安装卸载会很简单，于是使用 [huan/docker-wechat](https://github.com/huan/docker-wechat) 提供的 docker image，但是后来[输入时文本不可见的问题](https://github.com/huan/docker-wechat/issues/40)很恼人 ，也不知道怎么解决。

    注意到作者的 docker 是在 19.10 上构建的，在想会不会与我的 18.04 不够兼容，所以准备自己修改 docker，其实都已经 fork 好了，但是由于 [wine 对 18.04 的支持有个问题](https://forum.winehq.org/viewtopic.php?f=8&t=32192)，虽说可能跟输入法也不太有关，但是还是试着装这个，后面改写 docker file 时重新 build 总是出问题，一直没解决，所以决定放弃。

    于是就放弃 docker 了，想直接安装 wine，弊端似乎也就是卸载会有点繁，但是如果安装成功，那就用着呗，也不用卸载了。

    参考 [WeChat Desktop on Linux](https://ferrolho.github.io/blog/2018-12-22/wechat-desktop-on-linux)

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

??? tip "reboot wine"

    Sometimes, the wechat window freezes, i.e., the new messages cannot be updated. Try to reboot it,

    - use `wineboot`

    ```bash
    $ wineboot -s # shutdown
    $ wineboot -r # restart
    ```

    but it does not work, and it throws

    > 073c:fixme:font:get_name_record_codepage encoding 29 not handled, platform 1.

    - use `kill -s 9`

    kill the process of wechat, and then restart wechat, still not work

    - use `wineserver -k`

    it works! refer to [Ubuntu forum: Cant reboot wine](https://ubuntuforums.org/showthread.php?t=1451908)

    为了保留之前的所有设定及历史聊天记录，选择文件至 `/media/weiya/Seagate/WeChat`，但不要精确至 `/media/weiya/Seagate/WeChat/WeChat Files`

??? done "fixed: 窗口轮廓阴影"

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

!!! info
    ".desktop" 文件存放在 `~/.local/share/applications` 和 `/usr/share/applications`，而 `WeChat.desktop` 直接放在 `/home/weiya/Desktop`.

??? tip "修改 desktop 文件"

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

    后来跑去研究下 `wine` 的命令，想弄清楚那一长串 argument 是什么意思，才明白这应该是句 `wine start`，而 `/Unix` 是为了用 linux 格式的路径，详见 [3.1.1 How to run Windows programs from the command line](https://wiki.winehq.org/Wine_User%27s_Guide)，所以为了避免可能的转义问题，首先可以把 `wine C:\\\\windows\\\\command\\\\start.exe` 替换成 `wine start`，而空格转义还是用 `\ `，即最终 `WeChat.desktop` 文件为

    ```bash
    Exec=env WINEPREFIX="/home/weiya/.wine32" sh -c "wine start /Unix /home/weiya/.wine32/dosdevices/c:/ProgramData/Microsoft/Windows/Start\ Menu/Programs/WeChat/WeChat.lnk; python3 /home/weiya/disable-wechat-shadow.py"
    ```

    这个版本终于成功了！

    注意到 `env` 不要删掉，虽然在 `.bashrc` 中有设置，但是通过 desktop 启动时，并不会 source `.bashrc`，所以仍需保留这句设置，不然 `wine` 会找不到。

!!! tip
    一般地，`sh -c "command 1; command 2"`可以实现在 launcher 中同时运行两个命令，参考 [How to combine two commands as a launcher?](https://askubuntu.com/questions/60379/how-to-combine-two-commands-as-a-launcher)

??? tip "阴影窗口 id gap = 8 on T460P"

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

??? info "阴影窗口 id gap = 15 on XPS"
    在 XPS 上重复这一过程，发现 id gap 不再是 8。运行 `xwininfo`，然后将鼠标移至窗口边缘会出现十字，然后点击，便会返回当前阴影窗口 id。重复实验发现，此时阴影窗口 id 与主窗口 id 相差 15.

??? done "fixed: cannot send images"

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

??? note "DLL file"

    No clear idea about DLL file, such as `ole32.dll` suggested in [wine运行windows软件](https://jerry.red/331/wine%e8%bf%90%e8%a1%8cwindows%e8%bd%af%e4%bb%b6), this page， [Windows 7 DLL File Information - ole32.dll](https://www.win7dll.info/ole32_dll.html), might helps.

    And general introduction for DLL can be found in [DLL文件到底是什么，它们是如何工作的？](https://cloud.tencent.com/developer/ask/69913)

    - 类似 `.so`
    - 在Windows中，文件扩展名如下所示：静态库（`.lib`）和动态库（`.dll`）。主要区别在于静态库在编译时链接到可执行文件; 而动态链接库在运行时才会被链接。
    - 通常不会在计算机上看到静态库，因为静态库直接嵌入到模块（EXE或DLL）中。动态库是一个独立的文件。
    - 一个DLL可以在任何时候被改变，并且只在EXE显式地加载DLL时在运行时加载。静态库在EXE中编译后无法更改。一个DLL可以单独更新而无需更新EXE本身。

## Weylus

!!! info
    Post on 2022-06-13 15:40:10.

[Weylus](https://github.com/H-M-H/Weylus) turns tablet into a graphic tablet for PC, but never used, so remove it.

```bash
~$ apt list --installed | grep local
weylus/now 0.10.0 amd64 [installed,local]
```



## Xournal

!!! info
    Post on 2022-06-13 14:33:46

A hand note-taking software

Homepage: <https://github.com/xournalpp/xournalpp>

I have installed it, but never use it,

```bash
~$ apt list --installed | grep local
xournalpp/now 1.0.20-1~ubuntu18.04.1 amd64 [installed,local]
```

so just remove it.


## Zotero

??? tip "Click save button on which page?"

    - <https://www.sciencedirect.com>: press the save button on the page of the article, and no need to go to the Elsevier Enhanced Reader page by clicking "Download pdf", otherwise the saved item does not properly show the bib info and the type becomes webpage instead of journal

    ![](https://user-images.githubusercontent.com/13688320/114495190-3c6c9d80-9c50-11eb-9d24-dda8ee02acc7.png)

    ![](https://user-images.githubusercontent.com/13688320/114495254-560de500-9c50-11eb-8ab0-aa415b3608c0.png)

??? tip "sync to another cloud"

    The free size for saving is limited, and try to use a third-party cloud service for syncing. Use [坚果云](https://www.jianguoyun.com/), the configuration is as follows,

    ![](https://user-images.githubusercontent.com/13688320/124606874-e51d4880-de9f-11eb-85c7-71a1ebd51e44.png)

    Note that the password required in the left window can be obtained via clicking `Display Password` in the right window.
