# 笔记本作为服务器

有一台旧的笔记本电脑，Lenovo G40，刷了 Ubuntu 18.04，一直放在寝室吃灰，之前有段时间还试着将其当做周日在寝室办公的机器，但是相较于目前的 ThinkPad T460p，性能还是远不及的，打开软件的速度明显慢了很多。这次疫情待在家里，在摸索科学上网的几种途径后，觉得完全可以将其改成一台服务器。当然这个想法之前不是没有，但是因为想到没有 ip 地址，怎么能访问呢。当熟悉了 ssh 内网穿透以及 ngrok 这些工具之后，这些都变得不是事儿。

## SSH 远程登录

将系里服务器作为跳板机

```bash
weiya@G40 $ autossh -M 33000 -o "StrictHostKeyChecking=false" -o "ServerAliveInterval 10" -o "ServerAliveCountMax 3" -NR 30003:localhost:22 SID@SERVER
T460p $ ssh -L 30003:localhost:30003 SID@SERVER
T460p $ ssh -p 30003 weiya@127.0.0.1
```

复制的话也变得很简单了，

```bash
T460p $ ssh -P 30003 file weiya@127.0.0.1:
```

## 禁止合盖休眠

既然作为服务器了，没必要继续直接操作了，于是想合上盖子，但是会自动进入休眠状态（准确说是 suspend，而不是 hibernate），这时连接便都断开了。但是在设置界面也没找到直接关闭休眠的，最后在 [How to Change Lid Close Action in Ubuntu 18.04 LTS](https://tipsonubuntu.com/2018/04/28/change-lid-close-action-ubuntu-18-04-lts/) 中找到解决方案，

```bash
sudo vi /etc/systemd/logind.conf
```

commenting out `HandleLidSwitch=suspend` and changes it to `HandleLidSwitch=ignore`

最后需要 

```bash
systemctl restart systemd-logind.service
```

没想到这个也能使连接断开，于是需要继续连接一遍。


## 音乐播放器

目标：通过 ssh 远程打开音乐，但是仍在 G40 上播放，而不是像图象一样 forward 到本地播放。

本来一开始担心会像图象一样，需要通过类似 `-X` 这种选项来支持这种功能，如果支持了，下一步还要看看怎么直接在服务器端直接播放，而不是额外占用本地的资源来播放，要不然我干脆用本机的音乐播放器就完事了。

幸好，图象和音频不一样，如 [Playing a remote movie on the remote computer](https://unix.stackexchange.com/questions/76751/playing-a-remote-movie-on-the-remote-computer) 所说，

> Linux manages sound and display differently. You normally only get access to the screen if you've logged in locally, whereas sound is often available to all processes running on the system.

所以打消了我的顾虑，下一步便是直接播放了。刚好 G40 有一首下载好的歌曲，然后试了一下用 `aplay` 打开，但是非常嘈杂，根本不是音乐。

然后便下了个网易云客户端，本来以为音乐会在服务器端播放，但是没有任何声音，而且报错

> vlcpulse audio output error: PulseAudio server connection failure: Connection refused

再结合一下 `aplay` 的不正常播放，误以为 `PulseAudio` 出现了问题，所以按照 [PulseAudio server connection failure: Connection refused (debian stretch)](https://unix.stackexchange.com/questions/445386/pulseaudio-server-connection-failure-connection-refused-debian-stretch/567083) 操作一遍，似乎并没有什么问题。但是因为这个涉及到重启，所以提醒了我一个很重要的点，即自登陆自连接。

后来意识到 `aplay` 只能播放 `wav` 文件，而 `.mp3`需要其他的播放命令，比如 `mpg123`，参考 [How to play mp3 files from the command line?](https://askubuntu.com/questions/115369/how-to-play-mp3-files-from-the-command-line)

后来试了下 `sox`，声称可以支持多种格式，

```bash
$ sudo apt-get install sox
$ sudo apt-get install libsox-fmt-all
```

播放的界面很清爽但不简单，

![](music.gif)

其中 `-v 0.5` 调节音量。

!!! tip "音乐下载"
    - [超高无损音乐](https://www.sq688.com/)

先暂时写了个简单的列表顺序播放的脚本 

```bash
$ cat playmusic.sh 
#!/bin/bash
for music in ~/Music/*; do
	if [[ -f $music ]]; then
		play -v 0.35 "$music"
	fi
done
```

## 自登录自连接

因为重启后，一般会出现登录界面，需要输入用户名及密码。将 `/etc/gdm3/custom.conf` 中的这两行

```bash
#  AutomaticLoginEnable = true
#  AutomaticLogin = user1
```

改成 

```bash
AutomaticLoginEnable = true
AutomaticLogin = weiya
```

下一步还需要自动发起 ssh 连接至跳板服务器，这个可以再 `.profile` 中添加

```bash
$ echo "./autossh2ln001.sh" >> .profile
```

不过需要注意到如果存在 `.bash_profile` 或 `.bash_login`，则需要更改这些文件，因为只按顺序调用这三个的第一个。



