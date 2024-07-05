# 其它问题

## iframe

[Embedding a document inside another using the "iframe" tag](http://www.javascriptkit.com/howto/externalhtml.shtml)

## Variable tags

[Variable tags](https://help.shopify.com/themes/liquid/tags/variable-tags#capture)


## Track Non-JavaScript Visits In Google Analytics

[Track Non-JavaScript Visits In Google Analytics](https://www.simoahava.com/analytics/track-non-javascript-visits-google-analytics/)

## Ubuntu 搭建Apache

参考[How To Install the Apache Web Server on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04)


[How To Install Linux, Apache, MySQL, PHP (LAMP) stack on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-linux-apache-mysql-php-lamp-stack-on-ubuntu-16-04)

自定义根目录注意设置权限。如

[Apache2: 'AH01630: client denied by server configuration'](https://stackoverflow.com/questions/18392741/apache2-ah01630-client-denied-by-server-configuration)

## nginx虚拟主机配置

参考[How To Set Up Nginx Server Blocks (Virtual Hosts) on Ubuntu 14.04 LTS](https://www.digitalocean.com/community/tutorials/how-to-set-up-nginx-server-blocks-virtual-hosts-on-ubuntu-14-04-lts)

实现将本站用webhooks将其更新至阿里云服务器上，这与eslcn是同一个服务器，所以通过建立虚拟主机实现。


## URL 和 URI

参考[URL 和 URI 有什么不同? - 知乎](https://www.zhihu.com/question/19557151)

引用个人觉得简洁明了的回答

> URI (Identifier) 只讓你可以"區別"資源
> URL (Locator) 還讓你可以"找到"資源 (所以URL比一般的URI更"強", 是URI的子集)比如人的身份證號是 (非Locator的) Identifier: 不同身份證號一定是不同人, 但是用身份證號本身是找不到人的。这个人的住址或座标才是Locator。
>
> 作者：艾征霸
> 链接：https://www.zhihu.com/question/19557151/answer/130049112
> 来源：知乎
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## nginx配置跳转

比如将所有 `http://ServerIP/10 Boosting and Additive Trees/.*` 的访问301重定向到`http://$server_name/10-Boosting-and-Additive-Trees/10.1-Boosting-Methods/index.html`，在nginx配置文件中添加
```cmd
location ^~ '/10 Boosting and Additive Trees/' {
        rewrite ^/.* http://$server_name/10-Boosting-and-Additive-Trees/10.1-Boosting-Methods/index.html permanent;
}
```

几点说明：

1. 含等号时，不需要用`%20`进行编码，但需要加上引号，否则会报错，“invalid number of arguments in ‘location’ directive”，参考[nginx-rewrite-that-includes-a-blank-space](https://stackoverflow.com/questions/12101690/nginx-rewrite-that-includes-a-blank-spce)
2. 具体location匹配参考[nginx location 匹配规则](http://blog.csdn.net/wu5215080/article/details/55050858)
3. 参考[how-to-redirect-single-url-in-nginx](https://stackoverflow.com/questions/18037716/how-to-redirect-single-url-in-nginx)

## iframe跨域解决方案

参考[iframe跨域解决方案](http://blog.sina.com.cn/s/blog_63940ce201015w0d.html)

## flask + ajax + post + 跨域

参考[Flask 和 jQuery 联合实现传送 JSON 数据的 POST 跨域请求 (CORS)](https://zhuanlan.zhihu.com/p/27384232)

简单来说，

在客户端的 ajax 中添加

```javascripts
crossDomain: true,    // 允许跨域请求
```

另外，在服务器端，通过`CORS(app, resources=r'/*')`让flask服务器上所有的URL支持跨域请求。

另外，在ajax中post json的时候，注意用`JSON.stringify()`进行转换，否则格式为`否则格式为 a=2&b=3&now=14... `（参考[使用Ajax方式POST JSON数据包（转） - 平和的心 - 博客园](https://www.cnblogs.com/ajianbeyourself/p/5199144.html)）

## cookie vs. session

1. cookie 数据存放在客户的浏览器上，session 数据放在服务器上；
2. cookie 不是很安全，别人可以分析存放在本地的 COOKIE 并进行 COOKIE 欺骗，考虑到安全应当使用 session；
3. session 会在一定时间内保存在服务器上。当访问增多，会比较占用你服务器的性能。考虑到减轻服务器性能方面，应当使用 COOKIE；
4. 单个 cookie 在客户端的限制是3K，就是说一个站点在客户端存放的 COOKIE 不能超过 3K；

参考[理解Cookie和Session机制 - Andrew.Zhou - 博客园](https://www.cnblogs.com/andy-zhou/p/5360107.html)

## Travis CI

1. https://mwop.net/blog/2016-01-29-automating-gh-pages.html
2. [Deploying Docs on Github with Travis-CI](https://djw8605.github.io/2017/02/08/deploying-docs-on-github-with-travisci/)

注意 GITHUB_TOKEN 的设置，参考[Creating a personal access token for the command line](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)


## 图片旋转后无效果


原图片信息为

```bash
2018-10-29-cafe.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1080x1440, frames 3
```

采用 Shotwell 旋转图片后，
```bash
2018-10-29-cafe.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, Exif Standard: [TIFF image data, little-endian, direntries=3, orientation=lower-left, software=Shotwell 0.22.0], baseline, precision 8, 1080x1440, frames3
```

多了 EXIF 信息，其中有 orientation 信息。但是参考 [img tag displays wrong orientation](https://stackoverflow.com/questions/24658365/img-tag-displays-wrong-orientation/24658511) 的帖子，可知，有些浏览器并不遵循这个规则，即无视　EXIF 信息，从而网页端无效果。有人提到可以加上　

```css
img {
    image-orientation: from-image;
}
```

但这个似乎只有 Firefox 和　Safari 支持，Chrome 不支持。

解决方案：采用 `mogrify` 或 `convert` 进行旋转，如

```bash
mogrify -rotate "-90" 2018-10-29-cafe.jpg
```

查看文件信息

```bash
2018-10-29-cafe.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1440x1080, frames 3
```

注意到没有 EXIF 信息，而且 size 也由 1080x1440 变成了 1440x1080，所以这算是真旋转，而之前的加 EXIF 信息算是伪旋转吧。

如果文件已经存在 orientation 的 EXIF 信息，则 `mogrify -rotate` 似乎不起作用，解决方案为直接删掉 EXIF 信息，再进行相应的 rotate 操作，其中删除 EXIF 信息的命令是

```bash
exiftool -all= /tmp/my_photo.jpg
```

参考 [How can I read and remove meta (exif) data from my photos using the command line?](https://askubuntu.com/questions/260810/how-can-i-read-and-remove-meta-exif-data-from-my-photos-using-the-command-line)



## Image Slider

参考 [Image Slider - Jssor Slider](https://www.jssor.com/demos/image-slider.slider)

## 代码高亮

参考 [Jekyll 代码高亮的几种选择](https://blog.csdn.net/qiujuer/article/details/50419279)

## URL 中最后的斜杠

新建 tag 页面后，发现链接竟然跳到源站域名上去了，跟又拍那边的技术支持沟通也没找到原因，最后猛然想到是 tag 页面的 url 没有加斜杠，[查了一下](https://blog.csdn.net/u010525694/article/details/78591355)，加不加斜杠区别还挺大的。

## list start from 0

In `kramdown`, use an IAL declaration before the list, say

```html
{:start="3"}
1. test
1. test
1. test
```

Refer to [Support starting numbered lists with arbitrary number #211](https://github.com/gettalong/kramdown/issues/211)

Implementation for html can be found [here](https://stackoverflow.com/questions/15078393/begin-ordered-list-from-0-in-markdown).


## 字体选择

!!! note "少量字体直接用图片代替 2023-01-31 15:10:03"
    想把个人主页中 "Blog" 换成 "博客"，英文字体加载了 Abode 的艺术字体，但是并不支持中文。考虑到只有这两个中文字有需求，所以一种简单方式是直接生成这字体图片再插入进去。采用 [https://www.diyiziti.com/](https://www.diyiziti.com/) 上的字体，修改的 commit 详见 [:link:](https://github.com/szcf-weiya/szcf-weiya.github.io/commit/ed03cd9b379b4279a157665217cf5cea6eef8a42)

- [超赞！网页设计中最常见的30款英文字体](https://www.uisdc.com/30-west-typegraph-in-web-design)
- [Source Han Serif Simplified Chinese in Adobe Fonts](https://fonts.adobe.com/fonts/source-han-serif-simplified-chinese#fonts-section), and [Adobe Fonts](https://fonts.adobe.com/typekit)
- [Google Fonts](https://fonts.google.com/?category=Serif,Sans+Serif,Display,Monospace)
- [dafont.com](https://www.dafont.com/)
- [Webfont Generator](https://www.fontsquirrel.com/tools/webfont-generator)

## ruby 版本

今天 GitHub 提醒英文博客存在

```md
Known high severity security vulnerability detected in rubyzip < 1.3.0 defined in Gemfile.lock.
```

于是合并了它自动创建的 pull request: [Merge pull request #79 from szcf-weiya/dependabot/bundler/rubyzip-2.0.0](https://github.com/szcf-weiya/en/commit/54a7c509594211f7cc05736aa4adb5135bbe21d4)

但是后来发现在本地 `bundle exec jekyll serve` 预览失败。报错信息为

```md
rubyzip-2.0.0 requires ruby version >= 2.4, which is incompatible with the current version, ruby 2.3.1p112
```

看样子 ruby 版本不够，于是参考 [How do I upgrade to Ruby 2.2 on my Ubuntu system?](https://askubuntu.com/questions/839775/how-do-i-upgrade-to-ruby-2-2-on-my-ubuntu-system)

```bash
sudo apt-add-repository ppa:brightbox/ruby-ng
sudo apt-get update
sudo apt-get install ruby2.4
```

安装了 `ruby2.4`，完成后 `ruby` 也自动从 `ruby2.3` 更改到了 `ruby2.4`，但是重新运行

```bash
bundle exec jekyll serve
```

还是报同样的错误信息。后来参考 [Bundler using wrong ruby version](https://github.com/bundler/bundler/issues/4260)，运行

```bash
bundle env | grep ruby
```

发现里面的版本确实还是 2.3，于是按照里面的建议运行

```bash
gem install bundler
```

这样重新运行 `bundle env | grep ruby` 发现版本确实更新过来了。但是再次运行  

```bash
bundle exec jekyll serve
```

出现了新的错误


> An error occurred while installing commonmarker (0.17.13), and Bundler cannot continue.
> Make sure that `gem install commonmarker -v '0.17.13' --source 'https://rubygems.org/'` succeeds before bundling.

后来参考 [Error while installing json gem 'mkmf.rb can't find header files for ruby'](https://stackoverflow.com/questions/20559255/error-while-installing-json-gem-mkmf-rb-cant-find-header-files-for-ruby)

安装

```bash
sudo apt-get install ruby2.4-dev
```

这里按照报错信息检查发现 `/usr/lib/ruby/` 文件夹下的确没有 `include` 目录，只有 `2.4`, `2.5` 等目录，则这需要装对应 `ruby` 版本的 `dev`。当我把系统从 Ubuntu 16.04 更新到 Ubuntu 18.04 时，此时系统默认为 `ruby-2.5`,则我需要装的是

```bash
sudo apt-get install ruby2.5-dev
```

~~解决了问题！~~

发现还没有这么简单，总是报错

```
Could not find ffi-1.11.1 in any of the sources
```

但实际上已经装好了。不知道咋回事。

后来参考 https://jekyllrb.com/docs/ 重头开始，

```
gem install jekyll bundler
```

然后这一步一开始报错

```
ERROR:  Error installing jekyll:
        jekyll requires RubyGems version >= 2.7.0. Try 'gem update --system' to update RubyGems itself.
```

于是运行

```
[sudo] gem update --system
```

应该要加上 `sudo`，否则会报错

```
Installing RubyGems 3.0.6
ERROR:  While executing gem ... (Errno::EACCES)
    Permission denied @ dir_s_mkdir - /usr/local/lib/site_ruby
```

安装成功后，最后终于成功了！！

## Pygments

I have installed `pygmentize` and the corresponding python should be the system `python3`

```bash
$ which pygmentize 
/usr/bin/pygmentize
$ pygmentize -V
Pygments version 2.3.1, (c) 2006-2017 by Georg Brandl.
$ which python3
/usr/bin/python3
$ pip3 install Pygments
WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
To avoid this problem you can invoke Python with '-m pip' instead of running pip directly.
Looking in indexes: https://pypi.org/simple/
Requirement already satisfied: Pygments in /usr/lib/python3/dist-packages (2.3.1)
```

Pygments can be used in Jekyll via Rouge, see [Use Pygments for Code Snippet Highlighting in Jekyll - 李宇琨的博客](https://lyk6756.github.io/2016/11/22/use_pygments.html)

With such old version, the formatters like `material`, `github-dark` are not supported.

but the upgrade does not upgrade `pygmentize`, instead, it install it to `/home/weiya/.local/bin/pygmentize`

!!! warning "2022-10-29 21:14:49"
    However, upgrading `pygments` from 2.3.1 to 2.12.0 breaks the code block rendering in ESL-CN, see [:link:](https://github.com/szcf-weiya/techNotes/issues/25#issuecomment-1296036929). Locally, there is a conda env named `py36ESL`, in which `pygments` is installed via `pip` instead of `conda`. The solution is to install `pygments` via `conda install pygments==2.3.1` in `py36ESL` environment.

!!! tip "pygments vs rouge"
    `rouge` highlights contents within the block
    ```
    {% highlight LANG %}
    {% endhighlight %}
    ```
    It is much cumbersome to write these compared to three backsticks. But hopefully, css styles generated from pygments are also supported by rouge. So we do not need to worry about the highlighter.

## WeasyPrint

<https://weasyprint.org/>

- convert HTML to PDF: see also [:fontawesome-brands-zhihu:](https://zhuanlan.zhihu.com/p/343461128)
- customize HTML as PDF