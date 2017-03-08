# notes for emacs
## 常用命令

1. 切换缓存区：C-o
2. 水平新建缓存区：C-2
3. 垂直新建缓存区：C-3
4. 关闭当前缓存区：C-0
5. 删除缓存区：C-k
6. 只保留当前缓存区：C-1

## Emacs使用Fcitx中文

参考博客：[fcitx-emacs](http://wangzhe3224.github.io/emacs/2015/08/31/fcitx-emacs.html)

### Step 1: 确定系统当前支持的字符集

```bash
locale -a
```

若其中有zh_CN.utf8，则表明已经包含了中文字符集。

### Step 2: 设置系统变量

```bash
emacs ~/.bashrc
export LC_CTYPE=zh_CN.utf8 
source ~/.bashrc
```
