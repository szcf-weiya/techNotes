# vi 常用操作

## 复制
### 单行复制
在命令模式下，将光标移动到将要复制的行处，按“yy”进行复制；
### 多行复制
在命令模式下，将光标移动到将要复制的首行处，按“nyy”复制n行；其中n为1、2、3……

## 粘贴
    在命令模式下，将光标移动到将要粘贴的行处，按“p”进行粘贴

## vi复制多行文本的方法
### 方法1：
光标放到第6行，
输入：2yy
光标放到第9行，
输入：p
此方法适合复制少量行文本的情况，复制第6行（包括）下面的2行数据，放到第9行下面。
### 方法2：
命令行模式下输入
6,9 co 12
复制第6行到第9行之间的内容到第12行后面。
### 方法3：
有时候不想费劲看多少行或复制大量行时，可以使用标签来替代
光标移到起始行，输入ma
光标移到结束行，输入mb
光标移到粘贴行，输入mc
然后 :'a,'b co 'c   把 co 改成 m 就成剪切了
要删除多行的话，可以用 ：5, 9 de

## 去除BOM
[](https://segmentfault.com/q/1010000000256502)
vim 打开，
:set nobomb
:wq

## ctrl+s 假死

http://blog.csdn.net/tsuliuchao/article/details/7553003

使用vim时，如果你不小心按了 Ctrl + s后，你会发现不能输入任何东西了，像死掉了一般，其实vim并没有死掉，这时vim只是停止向终端输出而已，要想退出这种状态，只需按Ctrl + q 即可恢复正常。

## 执行当前脚本

参考[How to execute file I'm editing in Vi(m)](https://stackoverflow.com/questions/953398/how-to-execute-file-im-editing-in-vim)

另外也参考了[VIM中执行Shell命令（炫酷）](https://blog.csdn.net/bnxf00000/article/details/46618465)

## 打开另外一个文件

参考
1. [vim 打开一个文件后,如何打开另一个文件?](https://zhidao.baidu.com/question/873060894102392532.html)
2. [VI打开和编辑多个文件的命令 分屏操作 - David.Wei0810 - 博客园](https://www.cnblogs.com/david-wei0810/p/5749408.html)

## 删除光标后的字符 

```vi
d$
```

## write with sudo

For example, as said in [How does the vim “write with sudo” trick work?](https://stackoverflow.com/questions/2600783/how-does-the-vim-write-with-sudo-trick-work)

```bash
:w !sudo tee %
```

and such reference gives a more detailed explanation for the trick.