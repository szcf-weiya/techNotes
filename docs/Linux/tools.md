# Command-line Tools

## `awk`

### basic usage

`awk` 标准用法为

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

### processing two files

```bash
$ awk 'NR == FNR {# some actions; next} # other condition {# other actions}' file1.txt file2.txt
```

where

- `NR`: stores the total number of input records read so far, regardless of how many files have been read. 
- `FNR`: stores the number of records read from the current file being processed.
- so `NR == FNR` is only true when reading the first file
- `next` prevents the other condition/actions when reading the first file

!!! info
    A single number `1` can also serve as the condition, which means True, see also [awk '{sub(/pattern/, "foobar")} 1'](https://backreference.org/2010/02/10/idiomatic-awk/#comment-24936).
    > "1" is an always-true pattern; the action is missing, which means that it is `{print}` (the default action that is executed if the pattern is true).

refer to [Idiomatic awk](https://backreference.org/2010/02/10/idiomatic-awk/) (**A fantastic tutorial with many examples**)

For example, print the specific lines in `test.md` whose row numbers are defined in `line.txt`,

```bash
$ awk 'FNR==NR{wanted[$0]; next} FNR in wanted' lines.txt test.md
```

first came across in [selecting a large number of (specific) rows in file - Stack Overflow](https://stackoverflow.com/a/26672005), but it used `wanted[$0]++`, which does not make differences.

### sum of a column of numbers

```bash
awk '{s+=$1} END {print s}' data.txt
```

refer to [Bash command to sum a column of numbers - Stack Overflow](https://stackoverflow.com/questions/3096259/bash-command-to-sum-a-column-of-numbers) for other approaches.

For example, sum up the memory usage,

```bash
$ ps -e -o pid,cmd,%mem --sort=-%mem | awk 'NR > 1{s+=$NF} END {print s}'
```

### select lines with conditions

- select lines whose 2nd column is not empty: **usage of `!~`**

```bash
$ echo -e 'a \n c d' | awk '$2 !~ /^$/{print $2}'
d
```

- select lines whose 2nd column is not either empty or `-`: **usage of `|`**

```bash
$ echo -e 'a -\n c d' | awk '$2 !~ /-|^$/{print $2}'
d
```

- select lines whose both 2nd column and 3rd column are not empty: **usage of `&&`**

```bash
$ echo -e 'a - 1 \n c d 3 \n 5 6 ' | awk '$2 !~ /-|^$/ && $3 !~ /^$/ {print $2}'
d
```

## `cat`

- add text to the beginning of a file

```bash
echo 'task goes here' | cat - todo.txt > temp && mv temp todo.txt
```

where `-` represents the standard input, see also [:link:](https://blog.csdn.net/liuxiao723846/article/details/91041629)

alternatively, we can use `sed`, refer to [:link:](https://superuser.com/questions/246837/how-do-i-add-text-to-the-beginning-of-a-file-in-bash) for more details

## `column`

display the csv beautifully,

```bash
$ head file1.txt | column -s, -t
```

if two files share the same columns, 

```bash
$ (head file1.txt; head file2.txt) | column -s, -t
```

where

- `-s,`: specify the delimiter as `,`
- `-t`: print in a table

refer to [View tabular file such as CSV from command line](https://stackoverflow.com/questions/1875305/view-tabular-file-such-as-csv-from-command-line)

## `convert`

### 图片拼接

```bash
# 水平方向
convert +append *.png out.png
# 垂直方向
convert -append *.png out.png
```

参考 [How do I join two images in Ubuntu?](https://askubuntu.com/a/889772)

如果同时想两张图片高度一样，则加入 `-resize xW` 语法，如

```bash
# NOT work
$ convert map.png IMG_20210808_172104.jpg +append -resize x600 /tmp/p1.png
# work
$ convert +append map.png IMG_20210808_172104.jpg -resize x600 /tmp/p1.png
```

但是要注意此时 `+append` 要放在前面，也就是 `-resize` 需要紧跟着图片。

参考 [Merge Images Side by Side(Horizontally) - Stack Overflow](https://stackoverflow.com/questions/20737061/merge-images-side-by-sidehorizontally)

但是注意 `-resize` 会使得 orientation 无效，然后图片会发生旋转，参考 [ImageMagick convert rotates images during resize](https://legacy.imagemagick.org/discourse-server/viewtopic.php?t=33900)，使用 `-auto-orient` 参数，就能避免丢失图片中的 orientation 信息，

```bash
$ convert -auto-orient +append map.png IMG_20210808_172104.jpg -resize x600 /tmp/p2.png
```

### 缩小图片大小

```bash
# only specify the wide as 1024 pixel to keep the aspect ratio
convert input.png -resize 1024x out.png
convert input.png -quality 50% out.png
```

参考[How can I compress images?](https://askubuntu.com/questions/781497/how-can-i-compress-images)

### 合并jpg到pdf

参考[convert images to pdf: How to make PDF Pages same size](https://unix.stackexchange.com/questions/20026/convert-images-to-pdf-how-to-make-pdf-pages-same-size)

直接采用

```bash
pdftk A.pdf B.pdf cat output merge.pdf
```

得到的pdf中页面大小不一致，于是采用下面的命令

```bash
convert a.png b.png -compress jpeg -resize 1240x1753 \
                      -extent 1240x1753 -gravity center \
                      -units PixelsPerInch -density 150x150 multipage.pdf
```

注意重点是 `-density 150x150`，若去掉这个选项，则还是得不到相同页面大小的文件。

另外，上述命令是对于`.png`而言的，完全可以换成`.jpg`。

同时，注意`1240x1753`中间是字母`x`.

### pdf 转为 jpg

 `-quality 100` 控制质量
 `-density 600x600` 控制分辨率

并注意参数放置文件的前面

pdf 转 png 更好的命令是 `pdftoppm`，参考 [How to convert PDF to Image?](https://askubuntu.com/questions/50170/how-to-convert-pdf-to-image)

```bash
pdftoppm alg.pdf alg -png -singlefile
```

图片质量比 `convert` 好很多！！

### convert imgs to pdf

```bash
ls -1 ./*jpg | xargs -L1 -I {} img2pdf {} -o {}.pdf
pdftk likelihoodfree-design-a-discussion-{1..13}-1024.jpg.pdf cat output likelihoodfree-design-a-discussion.pdf
```

注意这里需要用 `ls -1`，如果 `ll` 则第一行会有 `total xxx` 的信息，即 `ll | wc -l` 等于 `ls -1 | wc -l` + 1，而且在我的 Ubuntu 18.04 中，`ll` 甚至还会列出

```bash
./
../
```

这一点在服务器上没看到。

### adjust brightness and contrast

!!! info
	Here is [one example](https://github.com/szcf-weiya/SZmedinfo/issues/5) used in my project.

```bash
$ convert -brightness-contrast 10x5 input.jpg output.jpg
```

where `10x5` increases the brightness 10 percent and the contrast 5 percent. 

These two values range from -100 to 100, and

- negative value: decrease
- zero: leave it off
- positive value: increase

more details refer to [ImageMagick: Annotated List of Command-line Options](https://www.imagemagick.org/script/command-line-options.php#brightness-contrast)

As an alternative, the GUI software `Shotwell` also provides similar functions, just clicking `enhance`.

## `cd`

- `cd "$(dirname "$0")"`: [cd current directory](https://stackoverflow.com/questions/3349105/how-to-set-current-working-directory-to-the-directory-of-the-script)

## `cp`

the common usage is `cp SOURCE DEST`, but if we want to copy multiple files into a single folder at once, we can use

```bash
cp -t DIRECTORY SOURCE
```

where `SOURCE` can be multiple files, inspired from [Copying multiple specific files from one folder to another - Ask Ubuntu](https://askubuntu.com/a/816826)

## `cut`

To select the first field of a file `file.txt`,

```shell
a=$(cut -d'.' -f1 <<< $1)_test
echo $a
```

where `-d'.'` is to define the delimiter, and then `-f1` get the first field.

If we need to get the last field, we can use `rev`, i.e.,

```bash
echo 'maps.google.com' | rev | cut -d'.' -f 1 | rev
```

refer to [How to find the last field using 'cut'](https://stackoverflow.com/questions/22727107/how-to-find-the-last-field-using-cut) and [10 command-line tools for data analysis in Linux](https://opensource.com/article/17/2/command-line-tools-data-analysis-linux)


## `date`

```bash
timestamp=$(date +"%Y-%m-%dT%H:%M:%S")
echo $timestamp
# 2020-02-11T10:51:42
```

we can compare two timestamps as follows

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

- `-d`: display time described by STRING, not 'now' (from `man date`) Alternatively, we can use format `-d "-10 days -8 hours -Iseconds"` to refer to the timestamp based on the current date, and `-Iseconds` specifies the unit as `seconds`, see the application in [Git: change-commit-time](../../Git/#change-commit-time).
- `+%[format-option]`: format specifiers (details formats refer to `man date`, but I am curious why `+`, no hints from `many date`, but here is one from [date command in Linux with examples](https://www.geeksforgeeks.org/date-command-linux-examples/))
- `-gt`: larger than, `-lt`: less than; with equality, `-ge` and `-le`, (from [Shell 基本运算符](https://www.runoob.com/linux/linux-shell-basic-operators.html))
- 条件表达式要放在方括号之间，并且要有空格, from [Shell 基本运算符](https://www.runoob.com/linux/linux-shell-basic-operators.html)

refer to [How to compare two time stamps?](https://unix.stackexchange.com/questions/375911/how-to-compare-two-time-stamps)

## `du`

- list size of subdirectories/files: `du -shc *`, where `-c` outputs the total

## `echo`

### string started with `-`

```bash
$ a="-n 1"
$ echo $a
1
$ echo "$a"
-n 1
```

double quotes are necessary, otherwise it would be treat as the option for `echo`. But if the string is pure `-`, the double quotes also failed,

```bash
$ b="-n"
$ echo "$b"
```

use `printf` would be more proper,

```bash
$ printf "%s\n" "$b"
-n
$ printf "%s\n" $b
-n
```

and no need to add double quotes.

refer to [Bash: echo string that starts with “-”](https://stackoverflow.com/questions/3652524/bash-echo-string-that-starts-with)

### print bytes

```bash
echo -n -e '\x66\x6f\x6f'
```

do not miss quotes, and `-e` is also necessary, refer to [echo bytes to a file](https://unix.stackexchange.com/questions/118247/echo-bytes-to-a-file)

### different save behavior

a column of elements would be stored in an array, then save via `echo` would result one line.

```bash
$ awk '{print $1}' duplicated.idx > t1.txt
$ cat t1.txt 
2
2
$ t1=$(awk '{print $1}' duplicated.idx)
$ echo $t1 > t2.txt
$ cat t2.txt 
2 2
```

## `ffmpeg`： 视频处理

### 去除音频

参考 [如何使用ffmpeg去除视频声音？](https://hefang.link/article/how-remove-voice-with-ffmpeg.html)

```bash
ffmpeg -i .\input.mp4 -map 0:0 -vcodec copy out.mp4
```

### 慢速播放和快速播放

```bash
# 2 times faster
$ ffmpeg -i input.mkv -filter:v "setpts=0.5*PTS" output.mkv
```

但是如果只对视频快速播放，而不处理音频，则文件的总时长仍不变。如果只关注视频，可以先去除音频，然后再做变速处理。

参考 [ffmpeg 视频倍速播放 和 慢速播放](https://blog.csdn.net/ternence_hsu/article/details/85865718)

除此之外，还可以通过先转成 raw bitstream 文件（未尝试），详见 [How to speed up / slow down a video – FFmpeg](https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video)

对于 GIF 文件，还可以用 `convert` 中的  `-delay` 选项实现。两者区别在于，前者会丢帧，而后者不会，

```bash
$ convert -delay 10 input.gif output.gif
```

refer to <https://infoheap.com/imagemagick-convert-edit-animated-gif-speed-fps/>

### 视频旋转

参考[How can I rotate a video?](https://askubuntu.com/questions/83711/how-can-i-rotate-a-video)

直接用

```bash
ffmpeg -i in.mov -vf "transpose=1" out.mov
```

然后报错 [“The encoder 'aac' is experimental but experimental codecs are not enabled”]((https://stackoverflow.com/questions/32931685/the-encoder-aac-is-experimental-but-experimental-codecs-are-not-enabled))

注意添加 `-strict -2` 要注意放置位置，一开始直接在上述命令后面加入，但失败，应该写成


```bash
ffmpeg -i in.mov -vf "transpose=1" -strict -2 out.mov
```

### 视频剪切

```bash
ffmpeg -ss 00:00:30.0 -i input.wmv -c copy -t 00:00:10.0 output.wmv
```

where

- (optional) `-ss` specifies the start timestamp, the format is `HH:MM:SS.xxx`
- (optional) `-t` specifies the duration, or use `-to` to specifies the end timestamp

refer to [Using ffmpeg to cut up video](https://superuser.com/questions/138331/using-ffmpeg-to-cut-up-video)

### concat

```bash
$ ffmpeg -f concat -safe 0 -i <(echo file $PWD/8xonlyVID_20210808_170208.mp4; echo file $PWD/8xonlyVID_20210808_170328.mp4) -c copy 8xonlyVID_20210808_170208+328.mp4
```

note that `$PWD` is necessary, otherwise it throws

> Impossible to open '/dev/fd/8xonlyVID_20210808_170328.mp4'
/dev/fd/63: No such file or directory

Also note that `&` seems will print the file info reversely,

```bash
$ echo "1" & echo "2"
[5] 5142
2
[4]   Done                    echo "1"
1
$ echo "1"; echo "2"
1
2
```

refer to [How to concatenate two MP4 files using FFmpeg? - Stack Overflow](https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg)

## `find`

```bash
$ find . -group group
$ find . -user user
```

refer to [list files with specific group and user name](https://unix.stackexchange.com/questions/518268/list-files-with-specific-group-and-user-name)

- [recursively list files in subdirectories](https://askubuntu.com/questions/307876/how-to-search-for-files-recursively-into-subdirectories)

```bash
$ find . -name '*.md'
```

alternatively,

```bash
$ ls -R | grep '\.md$'
```

where `.` needs to be escape and `$` is necessary, otherwise it would match strings like `rmd`.

- list all symbolic links

```bash
$ find . -type l -ls
```

!!! warning
	Parsing `ls`, such as `ls | grep "\->"` is a bad idea. [:link:](https://askubuntu.com/questions/522051/how-to-list-all-symbolic-links-in-a-directory)

## `grep`

- `-P`: perl-style regex
- `-o`: only print the matched part instead of the whole line
- `-v`: 反选

```bash
$ grep -oP "hello \K\w+" <<< "hello world"
world
```

where `\K` is the short form of `(?<=pattern)` as a zero-width look-behind assertion before the text to output, and `(?=pattern)` can be used as a zero-width look-ahead assertion after the text to output. For example, extract the text between `hello` and `weiya`.

```bash
$ grep -oP "hello \K(.*)(?=, weiya)" <<< "hello world, weiya!"
world
```

or equivalently,

```bash
$ grep -oP "(?<=hello )(.*)(?=, weiya)" <<< "hello world, weiya!"world
world
```

note that the space is also counted, 

```bash
$ grep -oP "(?<=hello)(.*)(?=, weiya)" <<< "hello world, weiya!"
 world
```

refer to [Can grep output only specified groupings that match? - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/13466/can-grep-output-only-specified-groupings-that-match)

!!! info
    [A practical example.](https://github.com/szcf-weiya/Cell-Video/blob/14a7dac4cd5c4bbfbf31d80eead85712eb8ba55a/report/update_bib.sh#L23)

- find all files given keywords, refer to [How do I find all files containing specific text on Linux? - Stack Overflow](https://stackoverflow.com/questions/16956810/how-do-i-find-all-files-containing-specific-text-on-linuxb)

```bash
grep -rnw '/path/to/somewhere/' -e 'pattern'
```

For example, J asked me about a situation that python failed to print to the log file in real time, and I indeed remembered that I had came cross this situation, but cannot find the relative notes. So I am trying to find files given possible keywords, such as `real time`, `print`, and finally I got the results

```bash
$ grep -rnw docs/*/*.md -e '输出'
docs/julia/index.md:765:> HASH函数是这么一种函数，他接受一段数据作为输入，然后生成一串数据作为输出，从理论上说，设计良好的HASH函数，对于任何不同的输入数据，都应该以极高的概率生成不同的输出数据，因此可以作为“指纹”使用，来判断两个文件是否相同。
docs/Linux/index.md:588:发现一件很迷的事情，要加上 `-u` 才能实现实时查看输出。
docs/shell/index.md:125:1. 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的；
```

As a comparison, the search function provided by GitHub is not so powerful since no related results returned in the search link <https://github.com/szcf-weiya/techNotes/search?q=%E8%BE%93%E5%87%BA&type=issues>

When I perform it on `syslog`, it did not return all matched results, and outputs,

```bash
$ grep -i failed syslog
Jul 24 13:17:11 weiya-ThinkPad-T460p gvfsd-metadata[13786]: g_udev_device_has_property: assertion 'G_UDEV_IS_DEVICE (device)' failed
Jul 24 14:02:53 weiya-ThinkPad-T460p gvfsd-metadata[13786]: g_udev_device_has_property: assertion 'G_UDEV_IS_DEVICE (device)' failed
Jul 24 14:02:53 weiya-ThinkPad-T460p gvfsd-metadata[13786]: g_udev_device_has_property: assertion 'G_UDEV_IS_DEVICE (device)' failed
Binary file syslog matches
```

refer to <https://stackoverflow.com/questions/23512852/grep-binary-file-matches-how-to-get-normal-grep-output>, add `-a` option.



## `htop`

A much more powerful command than `top`, refer to [Find out what processes are running in the background on Linux](https://www.cyberciti.biz/faq/find-out-what-processes-are-running-in-the-background-on-linux/)

- setup: modify the information layout when there are many CPUs. Firstly select the panel from the rightmost column, and then press `Left`, `Right`, `Up`, `Down` to move. The resulting configure file would be written into `~/.config/htop/htoprc`.

## `journalctl`

- list log of previous boots

```bash
$ journalctl --list-boots
```

- display last boot log

```bash
$ journalctl -b-1
```

see also: [:link:](https://askubuntu.com/questions/765315/how-to-find-previous-boot-log-after-ubuntu-16-04-restarts)

## `ln`

- with `-s`, create a soft link
- without `-s`, create a hard link

A "hard link" is actually between two directory entries; they're really the same file. And the number of the permission of `ll` also shows the number of hard links, such as `2` in `-rw-rw-r--  2`.

**the same file as another is they have the same inode number; no other file will have that.**

We can get the `inode` as follows,

```bash
$ stat resolve_utf.py | grep -i inode
Device: 811h/2065d	Inode: 14716809    Links: 2
```

refer to [How to find out a file is hard link or symlink?](https://unix.stackexchange.com/questions/170444/how-to-find-out-a-file-is-hard-link-or-symlink)

## `ls`

- `-S`: sort by filesize

### only show directory

```bash
ls -d */
```

refer to [Listing only directories using ls in Bash?](https://stackoverflow.com/questions/14352290/listing-only-directories-using-ls-in-bash)

My application: [TeXtemplates: create a tex template](https://github.com/szcf-weiya/TeXtemplates/blob/master/new.sh#L9)

### check whether a certain file type/extension exists in directory

```bash
if ls *.bib &>/dev/null; then
  #
fi
```

refer to [Check whether a certain file type/extension exists in directory](https://stackoverflow.com/questions/3856747/check-whether-a-certain-file-type-extension-exists-in-directory)

My application: [TeXtemplates: create a tex template](https://github.com/szcf-weiya/TeXtemplates/blob/master/new.sh#L18-L20)

## `mkdir`

- `mkdir -p`: [mkdir only if a dir does not already exist?](https://stackoverflow.com/questions/793858/how-to-mkdir-only-if-a-dir-does-not-already-exist)

## `mv`

### mv files with `xargs`

use `-I {}` to replace some str.

```bash
ls | grep 'config[0-9].txt' | xargs -I {} mv {} configs/
```

see more details in [mv files with | xargs](https://askubuntu.com/questions/487035/mv-files-with-xargs)

see also: [xargs命令_Linux xargs 命令用法详解：给其他命令传递参数的一个过滤器](http://man.linuxde.net/xargs)

## `paste`: 文本文件拼接

```bash
### 按列
paste file1 file2 > outputfile
### 按行
cat file1 file2 > outputfile
```

!!! note "See also"
	`cat`

convert a column to a row with delimiter `,`

```bash
$ for i in {1..10}; do echo $i; done | paste -s -d','
1,2,3,4,5,6,7,8,9,10
```

where `-s` aims to paste one file at a time instead of in parallel, which results in one line. Refer to [how to concatenate lines into one string](https://stackoverflow.com/questions/10303600/how-to-concatenate-lines-into-one-string)

## `pdftk`

- split pdf pages: `pdftk all.pdf cat 1 output first.pdf`, see also [arXiv](../TeX/arxiv.md).
- modify pdf metadata via `pdftk`

```bash
pdftk input.pdf dump_data output metadata
# edit metadata
pdftk input.pdf update_info metadata output output.pdf
```

## `ps2pdf, pdf2ps`

### reduce pdf file size

It can be used to reduce the pdf size. 

Generally, there are two major reasons why PDF file size can be unexpectedly large (refer to [Understanding PDF File Size](https://www.evermap.com/PDFFileSize.asp)).

- one or more fonts are stored inside PDF document.
- using images for creating PDF file.

I just got a large non-scanned pdf with size 136M, and it probably is due to many embedded fonts which can be checked in the properties.

Then I tried the command `ps2pdf` mentioned in [Reduce PDF File Size in Linux](https://www.journaldev.com/34668/reduce-pdf-file-size-in-linux), the file size is significantly reduced, only 5.5M!

```bash
$ ps2pdf -dPDFSETTINGS=/ebook Puntanen2011_Book_MatrixTricksForLinearStatistic.pdf Puntanen2011_Book_MatrixTricksForLinearStatistic_reduced.pdf
$ pdfinfo Puntanen2011_Book_MatrixTricksForLinearStatistic.pdf 
Creator:        
Producer:       Acrobat Distiller 8.0.0(Windows)
CreationDate:   Tue Jul 26 20:43:43 2011 CST
ModDate:        Fri Aug 19 19:57:50 2011 CST
Tagged:         no
UserProperties: no
Suspects:       no
Form:           AcroForm
JavaScript:     no
Pages:          504
Encrypted:      no
Page size:      439.37 x 666.142 pts
Page rot:       0
File size:      142394146 bytes
Optimized:      no
PDF version:    1.3
$ pdfinfo Puntanen2011_Book_MatrixTricksForLinearStatistic_reduced.pdf 
Creator:        
Producer:       GPL Ghostscript 9.26
CreationDate:   Tue Apr 13 18:04:57 2021 CST
ModDate:        Tue Apr 13 18:04:57 2021 CST
Tagged:         no
UserProperties: no
Suspects:       no
Form:           none
JavaScript:     no
Pages:          504
Encrypted:      no
Page size:      439.37 x 666.14 pts
Page rot:       0
File size:      5766050 bytes
Optimized:      no
PDF version:    1.4
```

We can compare the fonts before/after reducing,

```bash
$ pdffonts Puntanen2011_Book_MatrixTricksForLinearStatistic_reduced.pdf | wc -l
57
$ pdffonts Puntanen2011_Book_MatrixTricksForLinearStatistic.pdf | wc -l
125
```

and it seems not directly to remove fonts. Instead, most font names have been modified. Besides, these are duplicated font names (column one), such as 

```bash
$ pdffonts Puntanen2011_Book_MatrixTricksForLinearStatistic.pdf | sed 1,2d - | awk '{print $1}' | sort | uniq -c
      1 AMJQSV+LMSans8-Regular
        ...
     13 OELTPO+LMMathItalic10-Regular
        ...
      1 Times
      2 TimesNewRoman
      1 TimesNewRoman,Italic
      3 Times-Roman
        ...
     15 YCQSHP+LMRoman10-Bold
      4 YWGCMO+LMMathSymbols7-Regular
     16 ZMWYHT+LMRoman10-Regular
```

Count the number of unique names,

```bash
$ pdffonts Puntanen2011_Book_MatrixTricksForLinearStatistic.pdf | sed 1,2d - | awk '{print $1}' | sort | uniq -c | wc -l
50
$ pdffonts Puntanen2011_Book_MatrixTricksForLinearStatistic_reduced.pdf | sed 1,2d - | awk '{print $1}' | sort | uniq -c | wc -l
55
```

it shows that the reduced pdf does not have duplicated font names, (here the first two lines are removed, 57 = 55 + 2).

!!! tip "pdffonts"
    Another application of `pdffonts` is to check if the font has been embedded. If not, it might cause some display issue, such as the non-embedded `Symbol` in [The PDF viewer 'Evince' on Linux can not display some math symbols correctly](https://stackoverflow.com/questions/10277418/the-pdf-viewer-evince-on-linux-can-not-display-some-math-symbols-correctly). In contrast, Adobe Reader already ships with application-embedded instances of some fonts, such as `Symbol`, so it can render the pdf properly.
    A remedy is to use `gs`, see more details in the above reference.

### flatten pdf file

Flattening a PDF means to merge separated contents of the document into one so that,

- Interactive elements in PDF forms such as checkboxes, tex boxes, radio buttons, drop-down lists are no longer fillable
- Annotations become "native text"
- Multiple layers of text, images, page numbers, and header styles turn into one single layer.

An easy way is

```bash
pdf2ps orig.pdf - | ps2pdf - flattened.pdf
```

some alternatives can be found in [is-there-a-way-to-flatten-a-pdf-image-from-the-command-line](https://unix.stackexchange.com/questions/162922/is-there-a-way-to-flatten-a-pdf-image-from-the-command-line).

## `rename`

Ubuntu 18.04 和 CentOS 7 中的 `rename` 不一样，

```bash
# Ubuntu 18.04
$ rename -V
/usr/bin/rename using File::Rename version 0.20
# CentOS 7
$ rename -V
rename from util-linux 2.23.2
```

用法也有差异，前者采用类似 `sed` 格式语句进行替换

```bash
rename -n 's/Sam3/Stm32/' *.nc　　/*确认需要重命名的文件*/
rename -v 's/Sam3/Stm32/' *.nc　　/*执行修改，并列出已重命名的文件*/
```

而后者需要将替换的字符串当作参数传入，并且只替换第一次出现的字符串，即

```bash
rename Sam3 Stm32 *.nc
```

参考

- [Ubuntu中rename命令和批量重命名](http://www.linuxidc.com/Linux/2016-11/137041.htm)
- [Modifying replace string in xargs](https://stackoverflow.com/questions/10803296/modifying-replace-string-in-xargs)

## `sed`

- 打印特定行，比如第 10 行：`sed '10!d' file.txt`, 参考 [Get specific line from text file using just shell script](https://stackoverflow.com/questions/19327556/get-specific-line-from-text-file-using-just-shell-script)
- 打印行范围，`sed -n '10,20p' file.txt`，则单独打印第 10 行也可以由 `sed -n '10p' file.txt` 给出，如果采用分号 `;` 则不是连续选择，而只是特定的行，参考 [sed之打印特定行与连续行](https://blog.csdn.net/raysen_zj/article/details/46761253)
    - 第一行到最后一行：`sed -n '1,$p'`
    - 第一行和最后一行：`sed -n '1p;$p'`, not ~~`sed -n '1;$p'`~~
- 删除最后一行：`sed -i '$ d' file.txt`
- 在 vi 中注释多行：按住 v 选定特定行之后，按住 `:s/^/#/g` 即可添加注释，取消注释则用 `:s/^#//g`. 另见 VI.
- print lines between two matching patterns ([:material-stack-overflow:](https://unix.stackexchange.com/questions/264962/print-lines-of-a-file-between-two-matching-patterns)): `/^pattern1/,/^pattern2/p`, and if one want to just print once, use `/^pattern1/,${p;/^pattern2/q}`
- insertion (refer to [:link:](https://fabianlee.org/2018/10/28/linux-using-sed-to-insert-lines-before-or-after-a-match/) and [:link:]((https://www.thegeekstuff.com/2009/11/unix-sed-tutorial-append-insert-replace-and-count-file-lines/)))
    - insert before the line of matched expression: `sed '/expr/i something-to-insert'`
    - insert after the line: replace `i` with `a`
    - insert multiple lines: add `\n` in the text to insert
- 竖线 `|` 元字符是元字符扩展集的一部分，用于指定正则表达式的联合。如果某行匹配其中的一个正则表达式，那么它就匹配该模式。 [:link:](https://github.com/szcf-weiya/SZmedinfo/blob/af354f207d18ea270408562ac409b636eb17b5af/src/patterns_cut.sed#L4)
- directly replace hex string, such as [`'s/\xee\x81\xab/合/g'`](https://github.com/szcf-weiya/SZmedinfo/blob/af354f207d18ea270408562ac409b636eb17b5af/src/patterns_cut.sed#L6), see also [:link:](https://stackoverflow.com/questions/7760717/hex-string-replacement-using-sed)

Refer to

1. [Linux sed 命令用法详解：功能强大的流式文本编辑器](http://man.linuxde.net/sed)
2. [sed &amp; awk常用正则表达式 - 菲一打 - 博客园](https://www.cnblogs.com/nhlinkin/p/3647357.html)

### `-r`: 扩展的正则表达式

参考[Extended regexps - sed, a stream editor](https://www.gnu.org/software/sed/manual/html_node/Extended-regexps.html)

摘录如下

> The only difference between basic and extended regular expressions is in the behavior of a few characters: ‘?’, ‘+’, parentheses, and braces (‘{}’). While basic regular expressions require these to be escaped if you want them to behave as special characters, when using extended regular expressions you must escape them if you want them to match a literal character.

**就是说 basic 模式下，要使用特殊字符（如正则表达式中）需要转义，但 extended 模式相反，转义后表达的是原字符。**

举个例子

1. `abc?` becomes `abc\?` when using extended regular expressions. It matches the literal string ‘abc?’.
2. `c\+` becomes `c+` when using extended regular expressions. It matches one or more ‘c’s.
3. `a\{3,\}` becomes `a{3,}` when using extended regular expressions. It matches three or more ‘a’s.
4. `\(abc\)\{2,3\}` becomes `(abc){2,3}` when using extended regular expressions. It matches either `abcabc` or `abcabcabc`.
5. `\(abc*\)\1` becomes `(abc*)\1` when using extended regular expressions. Backreferences must still be escaped when using extended regular expressions.

### single or double quotes

When using double quotes, the string is first interpreted  by the shell before being passed to `sed`. As a result,

- more backslashes are needed (see also [my answer](https://askubuntu.com/a/1368043))

```bash
$ echo "\alpha" | sed 's/\\alpha/\\beta/'
\beta
$ echo "\alpha" | sed "s/\\\alpha/\\\beta/"
\beta
```

- command expressions (also dollar expressions) would be evaluated firstly

```bash
$ echo '`date`' | sed 's/`date`/`uptime`/'
`uptime`
$ echo '`date`' | sed "s/`date`/`uptime`/"
`date`
```

refer to [single quote and double quotes in sed - Ask Ubuntu](https://askubuntu.com/questions/1146789/single-quote-and-double-quotes-in-sed/)

## `sendmail`

send mail on the command line. On the stapc-WSL, install it via

```bash
$ sudo apt install sendmail
```

monitor the updates of `/var/log/apache2/error.log`. If the modified time is recent, then send email to alert.

```bash
while true; do
	last_date=$(date -r /var/log/apache2/error.log +%s)
#	curr_date=$(date -d "-1 mins" +%s)
	sleep 1m
	curr_date=$(date +%s)
	if [[ $last_date > $curr_date ]]; then
		(
			cat <<-EOT
			TO: ${TO}
			From: ${FROM}
			Subject: ${SUBJ}
			
			There is some updates on the /var/log/apache2/error.log
			
			EOT
		) | sendmail -v ${TO}
	fi
done
```

其中 `date -r file` 返回文件的上次修改时间，而 `+%s` 将时间转换为 seconds，方便进行比较，另外 `-d "-1 mins"` 能对时间进行加减处理。 

## `sort`

- [sort according to the third column](https://unix.stackexchange.com/questions/104525/sort-based-on-the-third-column): `sort -k 3,3 file.txt`
- [sort with header](https://stackoverflow.com/questions/14562423/is-there-a-way-to-ignore-header-lines-in-a-unix-sort)
: `cat your_data | (sed -u 1q; sort)`


## `tail`

- `-f`: output appended data as the file grows, powerful for checking the log file in real time.

## `tesseract`

OCR text extraction: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
$ tesseract tmp.png stdout -l eng+chi_sim quiet
```

where

- `quiet` redirects the warning message
- `stdout` directly outputs the results instead of writing into another text file

more details refer to `man tesseract`.

## `tmux`

可以实现本地终端分屏。

参考 [linux 工具——终端分屏与vim分屏](http://blog.csdn.net/u010454729/article/details/49496381)

!!! info
    现在改用 `Terminator`, 又称 `X-terminal-emulator`。

还可以切换后台运行，在服务器上操作特别方便。

!!! info
    此前还用过类似的软件，`screen`，

    ```bash
    screen -list #或screen -r
    screen -r [pid] # 进入
    ### ctrl+A, 然后输入":quit"
    ```

    更多用法详见 [linux screen 命令详解](https://www.cnblogs.com/mchina/archive/2013/01/30/2880680.html)，以及 [Kill detached screen session - Stack Overflow](https://stackoverflow.com/questions/1509677/kill-detached-screen-session)


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
# kill the current session
Ctrl + B, x
```

refer to
- [How do I access tmux session after I leave it?](https://askubuntu.com/questions/824496/how-do-i-access-tmux-session-after-i-leave-it)
- [Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)
- [tmux cheatsheet](https://gist.github.com/henrik/1967800)

## `type`

### `which` vs `type`

在 CentOS7 服务器上，

```bash
$ which -v
GNU which v2.20, Copyright (C) 1999 - 2008 Carlo Wood.
GNU which comes with ABSOLUTELY NO WARRANTY;
This program is free software; your freedom to use, change
and distribute this program is protected by the GPL.
```

`which` 可以返回 alias 中的命令，而且更具体地，`man which` 显示可以通过选项 `--read-alias` 和 `--skip-alias` 来控制要不要包括 alias. 

而在本地 Ubuntu 18.04 机器上，不支持 `-v` 或 `--version` 来查看版本，而且 `man which` 也很简单，从中可以看出其大致版本信息，`29 Jun 2016`。

那怎么显示 alias 呢，[`type` 可以解决这个问题](https://askubuntu.com/questions/102093/how-to-see-the-command-attached-to-a-bash-alias)，注意查看其帮助文档需要用 `help` 而非 `man`。

```bash
$ type scp_to_chpc 
scp_to_chpc is a function
scp_to_chpc () 
{ 
    scp -r $1 user@host:~/$2
}
```

## `uniq`

- count the frequency: `cat file.txt | sort | uniq -c`.
	- note that `sort` is necessary, otherwise `uniq` only performs locally
	- examples: [classificacaoFinal](https://github.com/szcf-weiya/TB/issues/46#issuecomment-831761174)

## `unzip`

### unzip all `.zip` file in a directory

tried `unzip *.zip` but does not work, it seems that I missed something although I have checked `man unzip` in which `*` is indeed allowed, then I found

```bash
unzip \*.zip
```

in [Unzip All Files In A Directory](https://stackoverflow.com/questions/2374772/unzip-all-files-in-a-directory/29248777)

Otherwise, use quotes `"*.zip"`. More advancely, only zip files with character `3`,

```bash
unzip "*3*.zip"
```

### `unzip` 和右键 `Extract Here` 的区别

对于 A.zip，假设内部结构为 `dir/file`，则通过 `unzip A.zip` 会直接得到 `dir/file`，而右键解压会得到 `A/dir/file`.

## `wget`

### wget a series of files in order

下载连续编号的文件，如

```
wget http://work.caltech.edu/slides/slides{01..18}.pdf
```

参考 [Wget a series of files in order](https://askubuntu.com/questions/240702/wget-a-series-of-files-in-order)

### `wget` vs `curl`

`wget` 不用添加 `-O` 就可以将下载的文件存储下来，但是 `curl` 并不默认将下载的文件存入本地文件，除非加上 `-o` 选项，而 `wget` 的 `-O` 只是为了更改文件名。

比如[这里](https://github.com/huan/docker-wine/blob/54e7ba2f042a59de72a06bafc37f1fb8c554541e/Dockerfile#L36)，直接将下载的内容输出到下一个命令

```bash
curl -sL https://dl.winehq.org/wine-builds/winehq.key | apt-key add -
```

更多比较详见 [What is the difference between curl and wget?](https://unix.stackexchange.com/questions/47434/what-is-the-difference-between-curl-and-wget)

## `unar`

如果 zip 文件解压乱码，可以试试 unar,

采用 `unar your.zip`

参考 [Linux文件乱码](https://www.findhao.net/easycoding/1605)

虽然它会自动识别编码，但有时候处理中文压缩文件仍然出现乱码，比如解压 <https://uploads.cosx.org/2011/03/SongPoem.tar.gz> 这个文件，

这时通过 `-e` 指定编码

```bash
 unar -e GB18030 ~/PDownloads/SongPoem.tar.gz 
/home/weiya/PDownloads/SongPoem.tar.gz: Tar in Gzip
  SongPoem.csv  (4171055 B)... OK.
  宋词.R  (583 B)... OK.
Successfully extracted to "SongPoem".
```

!!! info
    根据时间演化以及大小关系，常见中文编码方式

    > GB18030 > GBK > GB2312

    详见 [GB2312、GBK、GB18030 这几种字符集的主要区别是什么？](https://www.zhihu.com/question/19677619)

但是！这只能保证压缩文件的文件名以指定的编码格式进行编码，文件内容仍然是乱码，于是仍需指定编码格式。为了一劳永逸，直接转换成 UTF8 格式，

```bash
$ iconv -f GB18030 SongPoem.csv -t UTF8 -o SongPoem.csv.utf8
$ iconv -f GB18030 宋词.R -t UTF8 -o script.R
```