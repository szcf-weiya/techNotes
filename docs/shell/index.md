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

### another `for` loop

`start..stop..length`

```shell
for i in {5..50..5}; do
  echo $i
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

就是说 basic 模式下，要使用特殊字符（如正则表达式中）需要转义，但 extended 模式相反，转义后表达的是原字符。

举个例子

1. `abc?` becomes `abc\?` when using extended regular expressions. It matches the literal string ‘abc?’. 
2. `c\+` becomes `c+` when using extended regular expressions. It matches one or more ‘c’s. 
3. `a\{3,\}` becomes `a{3,}` when using extended regular expressions. It matches three or more ‘a’s. 
4. `\(abc\)\{2,3\}` becomes `(abc){2,3}` when using extended regular expressions. It matches either `abcabc` or `abcabcabc`.
5. `\(abc*\)\1` becomes `(abc*)\1` when using extended regular expressions. Backreferences must still be escaped when using extended regular expressions.

### 实战一

将

```
![IMG_0802](https://user-images.githubusercontent.com/13688320/72489850-733ae480-3850-11ea-8e51-15021588a7e6.jpg)
```

替换成

```
[IMG_0802]: https://user-images.githubusercontent.com/13688320/72489850-733ae480-3850-11ea-8e51-15021588a7e6.jpg
```

解决方案

```bash
sed -i "s/\!\[IMG_\([0-9]\{4\}\)\](\(.*\))/\[IMG_\1\]\: \2/g" FILENAME
```

- `\(\)` 用于匹配子串，并可以通过 `\1`, `\2` 引用
- `\!` 需要 escape
- `\2` 前面的空格不需要写成 `[ ]`，不然会直接出现 `[ ]`，而之前某次为了匹配多个空格需要写成 `[ ]*`

## awk

参考[技术|如何在Linux中使用awk命令](https://linux.cn/article-3945-1.html)

## 统计访问日志里每个 ip 访问次数

```bash
#!/bin/bash
cat access.log |sed -rn '/28\/Jan\/2015/p' > a.txt 
cat a.txt |awk '{print $1}'|sort |uniq > ipnum.txt
for i in `cat ipnum.txt`; do
    iptj=`cat  access.log |grep $i | grep -v 400 |wc -l`
    echo "ip地址"$i"在2015-01-28日全天(24小时)累计成功请求"$iptj"次，平均每分钟请求次数为："$(($iptj/1440)) >> result.txt
done
```

Refer to [用shell统计访问日志里每个ip访问次数](https://www.cnblogs.com/paul8339/p/6207182.html)

## split string while reading files

specify `IFS=`.

1. [How to split a tab-delimited string in bash script WITHOUT collapsing blanks?](https://stackoverflow.com/questions/19719827/how-to-split-a-tab-delimited-string-in-bash-script-without-collapsing-blanks) 
2. [Split String in shell script while reading from file](https://stackoverflow.com/questions/27500692/split-string-in-shell-script-while-reading-from-file)
3. [Read a file line by line assigning the value to a variable](https://stackoverflow.com/questions/10929453/read-a-file-line-by-line-assigning-the-value-to-a-variable) 

## distribute jobs into queues

since different queues has different quota, try to assign the job into available nodes.

```shell
queue=(bigmem large batch)
queues=()
for ((i=0;i<12;i++)) do queues+=(${queue[0]}); done;
for ((i=0;i<20;i++)) do queues+=(${queue[1]}); done;
for ((i=0;i<15;i++)) do queues+=(${queue[2]}); done;
```

refer to

- [Add a new element to an array without specifying the index in Bash](https://stackoverflow.com/questions/1951506/add-a-new-element-to-an-array-without-specifying-the-index-in-bash)
- [Repeat an element n number of times in an array](https://stackoverflow.com/questions/29205213/repeat-an-element-n-number-of-times-in-an-array)
- [The Double-Parentheses Construct](http://tldp.org/LDP/abs/html/dblparens.html)
- [Increment variable value by 1 ( shell programming)](https://stackoverflow.com/questions/21035121/increment-variable-value-by-1-shell-programming)
- [Shell 数组](http://www.runoob.com/linux/linux-shell-array.html)


## Command line arguments

refer to [Taking Command Line Arguments in Bash](https://www.devdungeon.com/content/taking-command-line-arguments-bash)

## join elements of an array in Bash

```shell
arr=(a b c)
printf '%s\n' "$(IFS=,; printf '%s' "${arr[*]}")"
# a,b,c
```

where `*` or `@` return all elements of such array.

refer to [How can I join elements of an array in Bash?](https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash)

### A more complex way

```shell
list=
for nc in {2..10}; do
  for nf in 5 10 15; do
    if [ -z "$list" ]
    then
        list=acc-$nc-$nf
    else
        list=$list,acc-$nc-$nf
    fi
  done
done
echo $list
```

## timestamp 

```shell
timestamp=$(date +"%Y-%m-%dT%H:%M:%S")
echo $timestamp
# 2020-02-11T10:51:42
```

### compare two timestamps

```shell
d1=$(date -d "2019-09-22 20:07:25" +'%s')
d2=$(date -d "2019-09-22 20:08:25" +'%s')
if [ $d1 -gt $d2 ]
then
  echo "d1 > d2"
else
  echo "d1 < d2"
fi
```

where 

- `-d`: display time described by STRING, not 'now' (from `man date`)
- `+%[format-option]`: format specifiers (details formats refer to `man date`, but I am curious why `+`, no hints from `many date`, but here is one from [date command in Linux with examples](https://www.geeksforgeeks.org/date-command-linux-examples/))
- `-gt`: larger than, `-lt`: less than; with equality, `-ge` and `-le`, (from [Shell 基本运算符](https://www.runoob.com/linux/linux-shell-basic-operators.html))
- 条件表达式要放在方括号之间，并且要有空格, from [Shell 基本运算符](https://www.runoob.com/linux/linux-shell-basic-operators.html)

refer to [How to compare two time stamps?](https://unix.stackexchange.com/questions/375911/how-to-compare-two-time-stamps)

## globbing for `ls` vs regular expression for `find`

Support we want to get `abc2.txt` as stated in [Listing with `ls` and regular expression
](https://unix.stackexchange.com/questions/44754/listing-with-ls-and-regular-expression),

`ls` does not support regular expressions, but it can work with globbing, or filename expressions. 

```shell
ls *[!0-9][0-9].txt
```

where `!` is complement.

Alternatively, we can use `find -regex`, 

```shell
find . -maxdepth 1 -regex '\./.*[^0-9][0-9]\.txt'
```

where

- `-maxdepth 1` disables recursive, and only to find files in the current directory

We also can add `-exec ls` to get the output of `ls`, and change the regex type by `-regextype egrep`.

## strip first 2 character from a string

simplest way:

```shell
${string:2}
```

some alternatives refer to [How can I strip first X characters from string using sed?](https://stackoverflow.com/questions/11469989/how-can-i-strip-first-x-characters-from-string-using-sed), or [Remove first character of a string in Bash](https://stackoverflow.com/questions/6594085/remove-first-character-of-a-string-in-bash)

## 