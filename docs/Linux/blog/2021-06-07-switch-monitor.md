---
comments: true
---

# Custom Shortcut to Switch External Display Mode

!!! info
    Post: [2021.06.07](https://github.com/szcf-weiya/techNotes/commit/19af9f4fb7ef0bc82798ebde3879e9e4c5ddbafe)

办公室电脑既可以作为显示屏，也可以在 PC 模式下使用 Windows 系统。在 PC 模式下，在 Ubuntu 上通过 synergy 共享键鼠，但是此时存在一个问题，因为 HDMI 仍然连着，所以在移动鼠标时中间有个 gap，也就是需要跳过外接显示屏才能移动到 PC。

试过在 synergy 中将 PC 机设置为 T460p 上方，这样移动鼠标只需往上，不过体验不是很好，而且 Ubuntu 顶端有状态栏而 PC 端底部也有task bar，移动时能明显感受到延时。另外一个策略便是切换显示屏 mode，由 joint 模式切换成 mirror。

注意到，当处于 mirror 模式下，eDP-1-1 primary 显示为 `1920x1080+0+0`，而如果是 joint mode，尺寸为 `1920x1080+1920+0`。受 [Swap between monitor display modes using shortcut](https://askubuntu.com/questions/958914/swap-between-monitor-display-modes-using-shortcut)
 启发，决定写脚本自定义快捷键

```bash
~$ cat switch_mirror_joint.sh 
#!/bin/bash
currentmode=$(xrandr -q | grep "primary 1920x1080+0+0")
if [[ -n $currentmode ]]; then
    #echo "mirror"
    xrandr --output HDMI-1-1 --left-of eDP-1-1 --transform none
else
    #echo "joint"
    xrandr --output HDMI-1-1 --same-as eDP-1-1 --scale-from 1920x1080
fi
```

!!! tip "--scale-from"
    如果两台显示器的分辨率不一致，则切换时会出现截断的问题。电脑屏幕分辨率为 1920x1080，而若外接显示器为 1920x1200，则切换至 mirror model 时，页面会以 1920x1200 为准，则在电脑屏幕上看不到底部 1200-1080 的部分。

    首先尝试过直接在设置中将外接显示器的分辨率改成 1920x1080，但是这会让画面模糊。

    另一种则是加上关键词 `--scale-from 1920x1080`，这样外接显示器的分辨率不会变，但会 scale 至电脑屏幕相同的分辨率大小。

    scale 之后再用 xrandr 查看便是 1920x1080，但在设置界面下显示器的分辨率仍保持不变。

!!! top "--transform none"
    另外观察到 scale 一次之后就不会再 scale 回来了。比如上面只在切换至 mirror 时有 scale，但是切换回 joint 时其 size 并不会变成 1920x1200，而继续保持 1920x1080. 为了避免这种情况，在转成 joint mode 时取消 scale，这可以通过 `--transform none` 实现。参考 [:link:](https://unix.stackexchange.com/questions/390099/reset-xrandr-or-switch-off-the-scale-from-setting-at-disconnect)

然后进入 keyboard shortcut 设置界面，

- Name: `switch display mode`
- Command: `/home/weiya/switch_mirror_joint.sh`
- Shortcut: `Ctrl+F7`

之所以选择 `F7` 是因为本身 F7 也支持切换 display mode，但是默认 external monitor 在右侧。试图直接更改 F7 的 binding commands，相关的 Ubuntu 官方帮助文档 [Keyboard](https://help.ubuntu.com/stable/ubuntu-help/keyboard.html.en) 及配置文件 [Custom keyboard layout definitions](https://help.ubuntu.com/community/Custom%20keyboard%20layout%20definitions)，但是无从下手。

!!! tip
    连接 HDMI 线没反应，也可以用 `xrandr` 排查是 PC 的问题还是显示器的问题。如果是电脑端没有显示连接，那很可能是显示器的问题。2022-09-16 初次连接 Yale 办公室的显示器时，便是电脑端没有插好。
