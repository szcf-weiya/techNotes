# Shell Notes

教程参考

1. [菜鸟教程](http://www.runoob.com/linux/linux-shell.html)
2. [GNU Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html#SEC_Contents)

## Special Parameters

- `$?`: 用于保存刚刚执行的命令的状态返回值
	- 0: 成功执行； 
	- 1-255: 表示执行失败
	- 2: All builtins return an exit status of 2 to indicate incorrect usage, generally invalid options or missing arguments.
	- 126: If a command is found but is not executable, the return status
	- 127: If a command is not found, the child process created to execute it returns
	- 128+N: When a command terminates on a fatal signal whose number is N, Bash uses the value 128+N as the exit status.
	- refer to [GNU Bash manual: 3.7.5 Exit Status](https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html) for more details.
- `$*`, `$@`: 引用参数。Expands to the positional parameters, starting from one.
- `$#`: 位置参数的个数
- `$0`: 脚本名称
- `$!`: the process ID of the job most recently placed into the background
- `$$`: the process ID of the shell

refer to

- [GNU Bash manual: 3.4.2 Special Parameters](https://www.gnu.org/software/bash/manual/html_node/Special-Parameters.html)

## Safer bash script

Came across some scripts [jperkel/nature_bash](https://github.com/jperkel/nature_bash/blob/e37ef568e81046207befaadd9872931c68a821ce/01_init.sh#L5) start with 

```bash
set -euo pipefail
```

or [docker-wechat/dochat.sh](https://github.com/huan/docker-wechat/blob/291c281df1ab6806156c286c7df1464b71eee2d1/dochat.sh#L11)

```bash
set -eo pipefail
```

both of which provide safeguards.

- `-e`: cause a bash script to exit immediately when a command fails
- `-o pipefail`: set the exit code of a pipeline to that of the rightmost command to exit with a non-zero status, or to zero if all commands of the pipeline exit successfully.
- `-u`: treat unset variables as an error and exit immediately

refer to

- [Safer bash scripts with 'set -euxo pipefail'](https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/)
- [解释bash脚本中set -e与set -o pipefail的作用](https://blog.csdn.net/t0nsha/article/details/8606886)

## shell变量

1. 定义变量时，变量名不加美元符号
2. 变量名和等号之间不能有空格
3. 变量名外面的花括号是可选的，加不加都行，加花括号是为了帮助解释器识别变量的边界，比如下面 `for` 循环中举的 `${skill}` 例子。

## iterators in `for` loop

- separated by space

```shell
for skill in Ada Coffe Action Java; do
    echo "I am good at ${skill}Script"
done
```


- `start..stop..length`

```shell
for i in {5..50..5}; do
  echo $i
done
```

!!! warning
    Not valid in `sh`, and use it with `bash`!
    ```bash
    $ ./sumbit_job.sh
    2
    4
    6
    8
    $ sh ./sumbit_job.sh
    {2..8..2}
    ```

- construct an array

actually, it can be used to construct an array,

```shell
arr=({1..10..2})
echo ${arr[@]}
for i in ${arr[@]}; do
  echo $i
done
```

- `seq`

alternatively, we can use `seq`,

```shell
for i in $(seq 5 5 50); do
    echo $i
done
```

## String

1. 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的；
2. 单引号字串中不能出现单引号（对单引号使用转义符后也不行）。
3. 双引号里可以有变量
4. 双引号里可以出现转义字符

### strip first 2 characters

simplest way:

```shell
${string:2}
```

some alternatives refer to [How can I strip first X characters from string using sed?](https://stackoverflow.com/questions/11469989/how-can-i-strip-first-x-characters-from-string-using-sed), or [Remove first character of a string in Bash](https://stackoverflow.com/questions/6594085/remove-first-character-of-a-string-in-bash)

a real application in my project:

```bash
list=""
for nc in {2..10}; do
  for nf in 5 10 15; do
    list="$list,acc-$nc-$nf"
    #https://stackoverflow.com/questions/6594085/remove-first-character-of-a-string-in-bash
    echo ${list:1}
  done
done
```

### strip from left or right

```bash
# 从左最大化匹配字符 `y`，然后截掉左边内容
$ var="xxyyzz" && echo ${var##*y}
zz
# 从左匹配第一个字符 `y`
$ var="xxyyzz" && echo ${var#*y}
yzz
# 从右最大化匹配字符 `y`，然后截掉右边内容
$ var="xxyyzz" && echo ${var%%y*}
xx
# 从右匹配第一个字符 `y`
$ var="xxyyzz" && echo ${var%y*}
xxy
```

### remove last character

with `bash 4.2+`,

```bash
$ var="xxyyzz" && echo ${var::-1}
xxyyz
```

refer to [Delete the last character of a string using string manipulation in shell script](https://unix.stackexchange.com/questions/144298/delete-the-last-character-of-a-string-using-string-manipulation-in-shell-script)

### replace character

```bash
$ var="xxyyzz" && echo ${var/xx/XX}
XXyyzz
```

### default value

- `${1:-foo}`: if parameter is unset or **null**, the expansion of word is substituted.
- `${1-foo}`: only substitute if parameter is unset.

refer to [How to write a bash script that takes optional input arguments?](https://stackoverflow.com/questions/9332802/how-to-write-a-bash-script-that-takes-optional-input-arguments)

applications:

- [Clouds/run_test_local.sh](https://github.com/szcf-weiya/Clouds/blob/b8abfc63078ac53e817c2a3d7f3e92d44cf47f61/run_test_local.sh#L5-L9)


## Array

1. 在Shell中，用括号来表示数组，数组元素用“空格”符号分割开。
2. 所有元素：`${ARRAY[@]}` 或者 `${ARRAY[*]}`
3. 数组长度：`${#ARRAY[@]}`
4. 从 0 编号：`${ARRAY[0]}`，类似 C 语言，与 `${ARRAY}` 返回结果相同。

## sed

参考

1. [sed命令_Linux sed 命令用法详解：功能强大的流式文本编辑器](http://man.linuxde.net/sed)
2. [sed &amp; awk常用正则表达式 - 菲一打 - 博客园](https://www.cnblogs.com/nhlinkin/p/3647357.html)

- 打印特定行，比如第 10 行：`sed '10!d' file.txt`, 参考 [Get specific line from text file using just shell script](https://stackoverflow.com/questions/19327556/get-specific-line-from-text-file-using-just-shell-script)
- 打印行范围，`sed -n '10,20p' file.txt`，则单独打印第 10 行也可以由 `sed -n '10p' file.txt` 给出，如果采用分号 `;` 则不是连续选择，而只是特定的行，参考 [sed之打印特定行与连续行](https://blog.csdn.net/raysen_zj/article/details/46761253)
    - 第一行到最后一行：`sed -n '1,$p'`
    - 第一行和最后一行：`sed -n '1p;$p'`, not ~~`sed -n '1;$p'`~~
- 删除最后一行：`sed -i '$ d' file.txt`
- 在 vi 中注释多行：按住 v 选定特定行之后，按住 `:s/^/#/g` 即可添加注释，取消注释则用 `:s/^#//g`. 另见 VI.
- print lines between two matching patterns ([:material-stack-overflow:](https://unix.stackexchange.com/questions/264962/print-lines-of-a-file-between-two-matching-patterns)): `/^pattern1/,/^pattern2/p`, and if one want to just print once, use `/^pattern1/,${p;/^pattern2/q}`
- insertion: [https://fabianlee.org/2018/10/28/linux-using-sed-to-insert-lines-before-or-after-a-match/](https://fabianlee.org/2018/10/28/linux-using-sed-to-insert-lines-before-or-after-a-match/) and [https://www.thegeekstuff.com/2009/11/unix-sed-tutorial-append-insert-replace-and-count-file-lines/](https://www.thegeekstuff.com/2009/11/unix-sed-tutorial-append-insert-replace-and-count-file-lines/)

### `|`的作用

> 竖线(|)元字符是元字符扩展集的一部分，用于指定正则表达式的联合。如果某行匹配其中的一个正则表达式，那么它就匹配该模式。 

### `-r`: 扩展的正则表达式

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

人总是善变的，过了一段时间，我又想把这些 img 下载到本地文件夹，但是之前处理过的文件都删掉了，只剩下传到 github 上的了，所以我首先要把文件下载到合适的位置并重命名。比如对于文件 `_posts/2019-12-21-quant-genetics.md`，只保留了 `https://user-images.githubusercontent.com/` 的链接，采用下面的脚本下载到合适的位置并重命名，

```bash
grep -E "https://user-images." _posts/2019-12-21-quant-genetics.md | 
	while read -a ADDR; do 
		if [ ${#ADDR[@]} -eq 2 ]; then 
			proxychains wget ${ADDR[1]} -O images/2019-12-21-quant-genetics/${ADDR[0]:1:8}.jpg; 
		fi; 
	done
```

其中

- `ADDR[0]:1:8` 是所谓的 ["Parameter Expansion" ${parameter:offset:length}](https://unix.stackexchange.com/questions/9468/how-to-get-the-char-at-a-given-position-of-a-string-in-shell-script)，用于提取特定范围的子串
- `wget -O` 是重命名，这里顺带移动到合适的位置
- `proxychains` 则是用于科学上网
- `read -a ADDR` 表示将分割后的字符串（比如默认按照空格进行分割，或者指定 `IFS=`）放进数组 ADDR 中，详见 `help read`，而 `man read` 并没有给出参数列表。另外需要注意到数组 `$ADDR` 返回结果为 `${ADDR[0]}`.
	- 读取单行文件时，采用 `;` 而非 pipeline，比如文件 `text.txt` 有单行内容 `1 2 3 4`. 应用 `read -a A < test.txt; echo ${A[0]}`，而非 `read -a A < test.txt | echo ${A[0]}`，后者返回空结果。


### 实战二

将 ESL-CN 中的手动输入的时间统一为发布的时间，详见[:octicons-commit-24:](https://github.com/szcf-weiya/ESL-CN/commit/431c8defa535b4448b8256a9639d5a633d00622d)

```bash
for file in $(find . -regex "./.*\.md"); do
    first=$(git log --date=short --format=%ad --follow $file | tail -1)
    echo $file $first
    sed -i "s/^\(|*\)[ ]*时间[ ]*|[^|]*\(|*\)[ ]*$/\1 发布 | $first \2/g" $file
done
```

其中 

- `$first` 提取最初 commit 的时间
- `\1` 和 `\2` 是为了处理有些表格写的是 `|--|--|`，而有些用的是 `--|--`，如果混用，则列会发生偏移，所以自适应保留原先的格式
- `|[^|]*` 是为了匹配第二个除表格符号 `|` 的内容，不要直接用 `|.*`，这样也会匹配最后的 `|`，从而 `\2` 匹配不到
- 定界符`^$` 为了防止匹配正文中的 `时间`

## 批量重命名

有时候下载文件时网站并没有区分同名文件，下载到本地后会出现 `A.zip` 与 `A (1).zip` 的情况，但这两个并不是相同的文件，所以避免以后误删，决定重命名。不过此类文件有好几个，批量处理代码为

```bash
$ ls -1 | grep "(1)" | while read -a ADDR; do mv "${ADDR[0]} (1).zip" "${ADDR[0]}_SOMETHING.zip"; done
```


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

其中 `awk` 标准用法为

```bash
awk '/pattern/ {print "$1"}'
```

`/pattern/` 可以是正则表达式，也可以是两个特殊的pattern，

- `BEGIN`: execute the action(s) before any input lines are read
- `END`: execute the action(s) before it actually exits

默认分隔符为空格，或者通过 `-F` 指定其它的分隔符，

```bash
$ echo 'a b' | awk '{print $2}'
b
$ echo 'a,b' | awk -F, '{print $2}'
b
```

其中 `$i` 指第 i 列，而 `$0` 指整条记录（including leading and trailing whitespace. ），详见 `man awk`.

另外上述脚本中用到了 `awk` 支持跨行状态的特性，即

```bash
$ echo -e 'a b \n c d' | awk '{print $2}'
b
d
```

其中 `-e` 是为了 escape 换行符，最后一列也可以用 `$NF` 表示，

```bash
$ echo -e 'a b \n c d' | awk '{print $NF}'
b
d
```

如果想要得到列数，则使用 `NF`,

```bash
$ echo -e 'a b \n c d' | awk '{print NF}'
2
2
```

如果默认每行列数相等，只想得到列数的话，可以使用

```bash
$ echo -e 'a b \n c d' | awk '{print NF; exit}'
2
```

如果数据文件为 `file.txt`，则可以直接用

```bash
awk '{print NF; exit}' file.txt
```


如果需要输出行数，则用

```bash
awk '{print NR, ":", $0}' file.txt
```

如果想从 0 开始，则改成 `NR-1`.

Refer to

- [用shell统计访问日志里每个ip访问次数](https://www.cnblogs.com/paul8339/p/6207182.html)
- [技术|如何在Linux中使用awk命令](https://linux.cn/article-3945-1.html)
- [unix - count of columns in file](https://stackoverflow.com/questions/8629330/unix-count-of-columns-in-file)
- [HANDY ONE-LINE SCRIPTS FOR AWK](http://www.pement.org/awk/awk1line.txt)
- [Learn How to Use Awk Special Patterns ‘BEGIN and END’ – Part 9](https://www.tecmint.com/learn-use-awk-special-patterns-begin-and-end/)
- [Print line numbers starting at zero using awk](https://stackoverflow.com/questions/20752043/print-line-numbers-starting-at-zero-using-awk)

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

where `((i++))` increases `i` by 1, and similar syntax can be

```bash
i=0
i=$((i+1))
# or
i=$(($i+1))
# or
((i+=1))
# or
((i++))
```

`((...))` also support general arithmetic operations, such as

```bash
a=30
b=10
echo $((a+=b))
echo $((a*=b))
echo $((a-=b))
echo $((a/=b))
echo $((a/=b))
# 40
# 400
# 390
# 39
# 3
```

but note that it does not allow float number, as the last equation, which should be `39/10=3.9`

the float calculation can take the advantage of other programs, such as

```bash
# Note that `BEGIN` cannot be removed, otherwise it is waiting for input file
# see the BEGIN and END pattern of awk
$ awk "BEGIN {print 39/10}"
3.9
$ bc <<< "39/10"
3
$ bc <<< "scale=2; 39/10"
3.90
```

refer to

- [Add a new element to an array without specifying the index in Bash](https://stackoverflow.com/questions/1951506/add-a-new-element-to-an-array-without-specifying-the-index-in-bash)
- [Repeat an element n number of times in an array](https://stackoverflow.com/questions/29205213/repeat-an-element-n-number-of-times-in-an-array)
- [The Double-Parentheses Construct](http://tldp.org/LDP/abs/html/dblparens.html)
- [Increment variable value by 1 ( shell programming)](https://stackoverflow.com/questions/21035121/increment-variable-value-by-1-shell-programming)
- [Shell 数组](http://www.runoob.com/linux/linux-shell-array.html)
- [How to do integer & float calculations, in bash or other languages/frameworks?](https://unix.stackexchange.com/questions/40786/how-to-do-integer-float-calculations-in-bash-or-other-languages-frameworks)

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

## Multiple `IFS`

```shell
while IFS= read -a ADDR; do
        IFS=':' read -a Line <<< $ADDR
        echo ${Line[0]};
done < <(grep -nE "finished" slurm-37985.out)
```

will also output the numbers of the finished line.

- `<()` is called [process substitution](https://superuser.com/questions/1059781/what-exactly-is-in-bash-and-in-zsh)
- `<<<` is known as `here string`, and [different from `<<`, `<`](https://askubuntu.com/questions/678915/whats-the-difference-between-and-in-bash)

refer to [How can I store the “find” command results as an array in Bash](https://stackoverflow.com/questions/23356779/how-can-i-store-the-find-command-results-as-an-array-in-bash)

my working case:

```shell
files=()
start_time=$(date -d "2019-09-21T14:11:16" +'%s')
end_time=$(date -d "2019-09-22T20:07:00" +'%s')
while IFS=  read -r -d $'\0'; do
  IFS='_' read -ra ADDR <<< "$REPLY"
  timestamp=$(date -d ${ADDR[2]} +'%s')
  if [ $timestamp -ge $start_time -a $timestamp -lt $end_time ]; then
    curr_folder="${ADDR[0]}_${ADDR[1]}_${ADDR[2]}"
    files+=("${ADDR[0]}_${ADDR[1]}_${ADDR[2]}")
    qsub -v folder=${curr_folder} revisit_sil_parallel.job
  fi
done < <(find . -maxdepth 1 -regex "\./oracle_setting_2019-09-.*recall\.pdf" -print0)
```

## 链接自动推送

```bash
find -regex "\./.*\.html" | sed -n "s#\./#https://esl.hohoweiya.xyz/#p" >> ../urls.txt
```

## Get path of the current script

we can get the path of the current script via

```bash
CURDIR=`/bin/pwd`
BASEDIR=$(dirname $0)
ABSPATH=$(readlink -f $0)
ABSDIR=$(dirname $ABSPATH)
```

refer to [darrenderidder/bashpath.sh](https://gist.github.com/darrenderidder/974638)

where `dirname`, together with `basename` aims to extract the filename and path, such as

```bash
$ basename /dir1/dir2/file.txt
file.txt
$ dirname /dir1/dir2/file.txt
/dir1/dir2
$ dirname `dirname /dir1/dir2/file.txt`
/dir1
```

but note that `dirname` would also return the parent directory of a directory, as shown in the last case.

Alternatively, we can use `${}` to extract the path,

```bash
# 从左开始最大化匹配到字符 `/`，然后截掉左边内容（包括`/`)
$ var=/dir1/dir2/file.txt && echo ${var##*/}
file.txt
# 文件后缀
$ var=/dir1/dir2/file.txt && echo ${var##*.}
txt
# 两个文件后缀（从左开始第一次匹配到字符 `.`，然后截掉左边内容（包括`/`)
$ var=/dir1/dir2/file.tar.gz && echo ${var#*.}
tar.gz
# 从右开始第一次匹配到字符 `/`
$ var=/dir1/dir2/file.txt && echo ${var%/*}
/dir1/dir2
# 从右最大化匹配到字符 `.`
$ var=/dir1/dir2/file.tar.gz && echo ${var%%.*}
/dir1/dir2/file
```

其中 `*` 表示要删除的内容，另外还需要一个字符表示截断点，注意到与 `#` 使用时截断字符在右边，而与 `%` 使用时截断字符在左边。

参考 [shell 提取文件名和目录名](http://blog.csdn.net/universe_hao/article/details/52640321)

## `if` statement

### `&> /dev/null`

We can add `&> /dev/null` to hidden the output information in the condition of `if`. For example, check if user exists,

```bash
--8<-- "docs/shell/if/if-id.sh"
```

note that for an existed user, the exit code is 0, while for a non-existed user, the exit code is non-zero, so the above command seems counter-intuitive.

```bash
~$ id weiya &> /dev/null 
~$ echo $?
0
~$ id weiya2 &> /dev/null 
~$ echo $?
1
```

another similar form can be found in [`>/dev/null 2>&1` in `if` statement](https://unix.stackexchange.com/questions/34491/dev-null-21-in-if-statement)

Another example: test if a file has an empty line,

```bash
--8<-- "docs/shell/if/if-empty-line.sh"
```

### logical operation

判断 Linux 发行版所属主流发行系列，另见 [Linux Distributions](../Linux/#linux-distributions)

```bash
--8<-- "docs/shell/if/if-dist.sh"
```

### check if uid equals gid

```bash
--8<-- "docs/shell/if/if-gid.sh"
```

note that use `-g` instead of `-G`, where the latter one would print all group ids.

### sum parameters

```bash
--8<-- "docs/shell/if/sum-paras.sh"
```

alternatively,

```bash
--8<-- "docs/shell/if/sum-paras2.sh"
```

where `shift` alternates the parameters such that `$1` becomes the next parameter.

The results are

```bash
$ ./sum-paras.sh 1 2 3 4
10
$ ./sum-paras2.sh 1 2 3 4
10
```

### test string

- `=~`: 判断左边的字符串是否能被右边的模式所匹配
- `-z $A`: 字符串长度是否为 zero（为空则为真，否则为假）
- `-n $A`: 字符串长度是否为 nonzero（为空则为假，否则为真）
- more details refer to `man test`.

```bash
--8<-- "docs/shell/if/if-sh.sh"
```

### test file

格式为 `[ EXPR FILE ]`，其中常见 `EXPR` 有

- `-f`: 测试其是否为普通文件，即 `ls -l` 时文件类型为 `-` 的文件
- `-d`: 测试其是否为目录文件，即 `ls -l` 时文件类型为 `d` 的文件
- `-e`: 测试文件是否存在；存在为真，否则为假
- `-r`: 测试文件对当前用户来说是否可读
- `-w`: 测试文件对当前用户来说是否可写
- `-x`: 测试文件对当前用户来说是否可执行
- `-s`: 测试文件大小是否不空，不空则真，空则假

例子

```bash
if [ ! -e /tmp/test ]; then
  mkdir /tmp/test
fi
```

refer to [bash条件判断之if语句](https://blog.51cto.com/64314491/1629175)

## `[` (aka `test`) vs `[[`

Refer to [What is the difference between test, [ and [[ ?](http://mywiki.wooledge.org/BashFAQ/031)

Both are used to evaluate expressions, but

- `[[` works only in Korn shell, Bash, Zsh, and recent versions of Yash and busybox `sh`
- `[` is POSIX utilities (generally builtin)

![](test.png)

But there are some differences:

- no word splitting or glob expansion will be done for `[[`, i.e., many arguments need not be quoted, while `[` usually should be quoted
- parentheses in `[[` do not need to be escaped

also see

- [What do square brackets mean without the “if” on the left?](https://unix.stackexchange.com/questions/99185/what-do-square-brackets-mean-without-the-if-on-the-left)
- [Is double square brackets preferable over single square brackets in Bash?](https://stackoverflow.com/questions/669452/is-double-square-brackets-preferable-over-single-square-brackets-in-ba)

### `$[`

Refer to [What does a dollar sign followed by a square bracket mean in bash?](https://unix.stackexchange.com/questions/209833/what-does-a-dollar-sign-followed-by-a-square-bracket-mean-in-bash)

With `$`, `[` is also can be used for arithmetic expansion, such as

```bash
$ echo $[ $RANDOM % 2 ]
0 # 1
$ echo $[ 1+2 ]
3
```

and actually `$[` syntax is an early syntax that was deprecated in favor of `$((`, although it's not completely removed yet.

### `=` vs `==` vs `-eq`

from the above discussion:

- `==` is a bash-ism
- `=` is POSIX

In bash the two are equivalent, but in plain sh `=` is the only one guaranteed to work. And these two are for string comparisons, while `-eq` is for numerical ones.

refer to [Shell equality operators (=, ==, -eq)](https://stackoverflow.com/questions/20449543/shell-equality-operators-eq)

## compare string


## grep keep the first line (use `sed` instead)

Refer to [Include header in the 'grep' result](https://stackoverflow.com/questions/12920317/include-header-in-the-grep-result)

I am using

```bash
$ sinfo -o "%P %N %C %G" -N | grep gpu
```

to get the GPU status of the nodes on the cluster, but the header cannot be kept, then I tried

```bash
$ sinfo -o "%P %N %C %G" -N | { head -1; grep gpu; }
```

but it only shows the header

Next I got the excellent solution via `sed`,

```bash
$ sinfo -o "%P %N %C %G" -N | sed -n "1p;/gpu/p"
```

and it can hide the highlighter of `gpu`.

## compare two blocks in a txt file

for example, compare L82-95 with L108-123,

```bash
$ diff <(sed -n "82,95p" measure.jl) <(sed -n "108,123p" measure.jl)
```

## `||`

No `try...catch` in bash script, but `command1 || command2` can achieve similar functionality, if `command1` fails then `command2` runs.

An application is to create a new tmux session if there is no such a tmux session,

```bash
t() { tmux a -t $1 || tmux new -s $1; }
```

refer to [Is there a TRY CATCH command in Bash](https://stackoverflow.com/questions/22009364/is-there-a-try-catch-command-in-bash)

## Replace a multiple line string 

<!--
![](https://img.shields.io/badge/-sed-brightgreen) ![](https://img.shields.io/badge/-perl-orange)
-->

`sed` is powerful for replace in one-line, but it would be much more complicated for multiple line string. 

An alternative is to use `perl`,

```bash
$ printf "###BEGIN-EXCLUDE\nwww\n###END-EXCLUDE" | perl -0777 -pe "s/###BEGIN-EXCLUDE(.*?)###END-EXCLUDE/xxx/igs"
xxx
```

where

- `-0777` causes Perl to slurp files whole, see [-0[octal/hexadecimal]](https://perldoc.perl.org/perlrun) for more details
- `-e commandline`
- `-p`: my understanding is something for printing, but the [official document](https://perldoc.perl.org/perlrun) explain more 

For a file to be edited in-place, we add `-i`,

```bash
$ perl -0777 -i -pe "s/###BEGIN-EXCLUDE(.*?)###END-EXCLUDE//igs" test.txt
```

if we supply an extension such as `-i'.orig'`, then the old file would be backed up to `test.txt.orig`.

Refer to [How can I use sed to replace a multi-line string?](https://unix.stackexchange.com/questions/26284/how-can-i-use-sed-to-replace-a-multi-line-string)

Moreover, `.*?` is a non-greedy match, comparing to `.*`. 

```bash
$ echo "s12tAs34t" | perl -pe "s/s(.*?)t/xx/igs"
xxAxx
$ echo "s12tAs34t" | perl -pe "s/s(.*)t/xx/igs"
xx
```

Refer to [6.15. Greedy and Non-Greedy Matches docstore.mik.ua/orelly/perl/cookbook/ch06_16.htm](https://docstore.mik.ua/orelly/perl/cookbook/ch06_16.htm) for details.

By default, `grep` performs greedy match, adding option `-P` would enable Perl's non-greedy match.

![](https://user-images.githubusercontent.com/13688320/119338519-adc55480-bcc2-11eb-9fac-bb32d2b9c72d.png)

!!! info
	An real application: [update_bib.sh](https://github.com/szcf-weiya/Cell-Video/blob/67fef1b7737e97631aa9568b000a8c61ef1590f4/report/update_bib.sh#L21)

Refer to [How to do a non-greedy match in grep?](https://stackoverflow.com/questions/3027518/how-to-do-a-non-greedy-match-in-grep/3027524)
