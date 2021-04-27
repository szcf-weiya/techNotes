# Small Tools

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

## `htop`

A much more powerful command than `top`, refer to [Find out what processes are running in the background on Linux](https://www.cyberciti.biz/faq/find-out-what-processes-are-running-in-the-background-on-linux/)

## `paste, cat`: 文本文件拼接

```bash
### 按列
paste file1 file2 > outputfile
### 按行
cat file1 file2 > outputfile
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