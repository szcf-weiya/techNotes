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

refer to [Idiomatic awk](https://backreference.org/2010/02/10/idiomatic-awk/)

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

refer to [How to find the last field using 'cut'](https://stackoverflow.com/questions/22727107/how-to-find-the-last-field-using-cut)

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

## `ffmpeg`： 视频处理

### 去除音频

参考 [如何使用ffmpeg去除视频声音？](https://hefang.link/article/how-remove-voice-with-ffmpeg.html)

```bash
ffmpeg -i .\input.mp4 -map 0:0 -vcodec copy out.mp4
```

### 慢速播放和快速播放

参考 [ffmpeg 视频倍速播放 和 慢速播放](https://blog.csdn.net/ternence_hsu/article/details/85865718)

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

## `find`

```bash
$ find . -group group
$ find . -user user
```

refer to [list files with specific group and user name](https://unix.stackexchange.com/questions/518268/list-files-with-specific-group-and-user-name)

## `htop`

A much more powerful command than `top`, refer to [Find out what processes are running in the background on Linux](https://www.cyberciti.biz/faq/find-out-what-processes-are-running-in-the-background-on-linux/)

## `ls`

- `-S`: sort by filesize

### modify and change time

在找学习资料时，突然不是很确定当初是否已经在用这台笔记本了，所以想确定一下本机的装机时间，参考 [How can I tell what date Ubuntu was installed?](https://askubuntu.com/questions/1352/how-can-i-tell-what-date-ubuntu-was-installed)，主要时通过查看文件的上次修改时间，比如

```bash
$ ls -lt /var/log/installer/
total 1200
-rw-rw-r-- 1 root   root 464905 Dec  2  2016 initial-status.gz
-rw-r--r-- 1 root   root     60 Dec  2  2016 media-info
-rw------- 1 syslog adm  334743 Dec  2  2016 syslog
-rw------- 1 root   root   2467 Dec  2  2016 debug
-rw------- 1 root   root 407422 Dec  2  2016 partman
-rw------- 1 root   root     17 Dec  2  2016 version
-rw------- 1 root   root    956 Dec  2  2016 casper.log
```

又如

```bash
$ ls -lt /
...
drwxrwxr-x   2 root root       4096 Dec  2  2016 cdrom
drwx------   2 root root      16384 Dec  2  2016 lost+found
drwxr-xr-x   2 root root       4096 Apr 21  2016 srv
```

出现了 2016.04.21 的一条记录。但如果我加上 `-c`，结果竟然不一样

```bash
$ ls -clt /
...
drwxrwxr-x   2 root root       4096 Dec  2  2016 cdrom
drwxr-xr-x   2 root root       4096 Dec  2  2016 srv
drwx------   2 root root      16384 Dec  2  2016 lost+found
```

难道 `ls` 默认显示的时间不是上次修改时间吗？？另外注意到 `srv` 其实是一个空文件夹。

这时我用 `stat` 进一步查看，

```bash
$ stat /srv
  File: /srv
  Size: 4096      	Blocks: 8          IO Block: 4096   directory
Device: 825h/2085d	Inode: 1179649     Links: 2
Access: (0755/drwxr-xr-x)  Uid: (    0/    root)   Gid: (    0/    root)
Access: 2021-05-05 08:43:20.955106697 +0800
Modify: 2016-04-21 06:07:49.000000000 +0800
Change: 2016-12-02 02:46:47.363728274 +0800
 Birth: -
```

发现有两个修改时间，`Modify` 和 `Change`，[两者区别:material-stack-overflow:](https://unix.stackexchange.com/questions/2464/timestamp-modification-time-and-created-time-of-a-file)在于

- `Modify`: the last time the file was modified (content has been modified)
- `Change`: the last time meta data of the file was changed (e.g. permissions)

然后进一步查看 Windows 系统的时间，

```bash
$ ll -clt
...
drwxrwxrwx  1 weiya weiya       4096 Oct  1  2016 '$Recycle.Bin'/
drwxrwxrwx  1 weiya weiya          0 Sep 29  2016  FFOutput/
-rwxrwxrwx  2 weiya weiya   15151172 Jul  2  2016  WindowsDENGL.tt2*
-rwxrwxrwx  2 weiya weiya   16092228 Jul  2  2016  WindowsDENG.tt2*
-rwxrwxrwx  2 weiya weiya   16217976 Jul  2  2016  WindowsDENGB.tt2*
-rwxrwxrwx  1 weiya weiya     400228 Mar 19  2016  bootmgr*
-rwxrwxrwx  1 weiya weiya          1 Mar 19  2016  BOOTNXT*
drwxrwxrwx  1 weiya weiya       8192 Mar 18  2016  Boot/
```

最早可以追溯到 2016.03.18.

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

## `paste, cat`: 文本文件拼接

```bash
### 按列
paste file1 file2 > outputfile
### 按行
cat file1 file2 > outputfile
```

convert a column to a row with delimiter `,`

```bash
$ for i in {1..10}; do echo $i; done | paste -s -d','
1,2,3,4,5,6,7,8,9,10
```

where `-s` aims to paste one file at a time instead of in parallel, which results in one line. Refer to [how to concatenate lines into one string](https://stackoverflow.com/questions/10303600/how-to-concatenate-lines-into-one-string)

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
