# shell相关

教程参考

1. [菜鸟教程](http://www.runoob.com/linux/linux-shell.html)

## shell变量

1. 定义变量时，变量名不加美元符号
2. 变量名和等号之间不能有空格
3. 变量名外面的花括号是可选的，加不加都行，加花括号是为了帮助解释器识别变量的边界，比如下面这种情况：
```shell
for skill in Ada Coffe Action Java; do
    echo "I am good at ${skill}Script"
done
```

## shell字符串

1. 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的；
2. 单引号字串中不能出现单引号（对单引号使用转义符后也不行）。
3. 双引号里可以有变量
4. 双引号里可以出现转义字符

## shell数组

1. 在Shell中，用括号来表示数组，数组元素用“空格”符号分割开。

## sed用法

参考

1. [sed命令_Linux sed 命令用法详解：功能强大的流式文本编辑器](http://man.linuxde.net/sed)
2. [sed &amp; awk常用正则表达式 - 菲一打 - 博客园](https://www.cnblogs.com/nhlinkin/p/3647357.html)

## `|`的作用

> 竖线(|)元字符是元字符扩展集的一部分，用于指定正则表达式的联合。如果某行匹配其中的一个正则表达式，那么它就匹配该模式。

## `-r`的作用

也就是使用扩展的正则表达式

参考[Extended regexps - sed, a stream editor](https://www.gnu.org/software/sed/manual/html_node/Extended-regexps.html)

摘录如下

> The only difference between basic and extended regular expressions is in the behavior of a few characters: ‘?’, ‘+’, parentheses, and braces (‘{}’). While basic regular expressions require these to be escaped if you want them to behave as special characters, when using extended regular expressions you must escape them if you want them to match a literal character.

就是说basic模式下，要使用特殊字符（如正则表达式中）需要转义，但extended模式相反，转义后表达的是原字符。

举个例子

1. `abc?` becomes `abc\?` when using extended regular expressions. It matches the literal string ‘abc?’. 
2. `c\+` becomes `c+` when using extended regular expressions. It matches one or more ‘c’s. 
3. `a\{3,\}` becomes `a{3,}` when using extended regular expressions. It matches three or more ‘a’s. 
4. `\(abc\)\{2,3\}` becomes `(abc){2,3}` when using extended regular expressions. It matches either `abcabc` or `abcabcabc`.
5. `\(abc*\)\1` becomes `(abc*)\1` when using extended regular expressions. Backreferences must still be escaped when using extended regular expressions.

## awk

参考[技术|如何在Linux中使用awk命令](https://linux.cn/article-3945-1.html)