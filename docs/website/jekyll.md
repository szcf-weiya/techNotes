## Jekyll

Jekyll is a **Ruby Gem** that can be installed on most systems.

- Ruby: 一种开源的面向对象程序设计的服务器端脚本语言，在 20 世纪 90 年代中期由日本的松本行弘（まつもとゆきひろ/Yukihiro Matsumoto）设计并开发。[Ruby 教程](https://www.runoob.com/ruby/ruby-tutorial.html)
- [Jekyll is written in Ruby.](https://jekyllrb.com/docs/ruby-101/)
- Gem: Gems are code you can include in Ruby projects. Jekyll is a gem. Many Jekyll plugins are also gems.
- Gemfile: a list of gems.
- Bundler: a gem that installs all gems in `Gemfile`.
- RubyGems: Ruby 的一个包管理器，它提供一个分发 Ruby 程序和库的标准格式，还提供一个管理程序包安装的工具，类似于 Ubuntu 下的 apt-get, Centos 的 yum，Python 的 pip。[https://rubygems.org/](https://rubygems.org/)

### run locally

```bash
# 0. install prerequisites: https://jekyllrb.com/docs/installation/ubuntu/
# 1. install the jekyll and bundler gems
gem install jekyll bundler
# 2. build the site locally
bundle exec jekyll serve
```

if it broken down, such as 

```bash
/home/weiya/gems/gems/octokit-4.14.0/lib/octokit/middleware/follow_redirects.rb:14:in `<module:Middleware>': uninitialized constant Faraday::Error::ClientError (NameError)
```

try to [update the local gem](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll) via

```bash
bundle update github-pages
```

### unsupported plugins by Github pages

> Github Pages sites are generated using the `--safe` option to disable plugins (with the exception of some [whitelisted plugins](https://pages.github.com/versions/)) for security reasons.
> source: [Plugins on GitHub Pages](https://jekyllrb.com/docs/plugins/installation/)

If use third-party plugins, such as [gjtorikian/jekyll-last-modified-at](https://github.com/gjtorikian/jekyll-last-modified-at), there are two possible ways

- build locally
- use travis or [github actions](https://jekyllrb.com/docs/continuous-integration/github-actions/).

refer to [How do I configure GitHub to use non-supported Jekyll site plugins?](https://stackoverflow.com/questions/28249255/how-do-i-configure-github-to-use-non-supported-jekyll-site-plugins)


## Jekyll Part 13: Creating an Article Series

[Jekyll Part 13: Creating an Article Series](https://digitaldrummerj.me/blogging-on-github-part-13-creating-an-article-series/)

## Add an "Updated" field to Jekyll posts

[Add an "Updated" field to Jekyll posts](https://zzz.buzz/2016/02/13/add-an-updated-field-to-your-jekyll-site/)

## 博客中插入网易云音乐

这个很容易实现，只需要在网易云中搜索要插入的音乐，然后点击“生成外链播放器”，将iframe代码插入博客的相应位置。

比如，我想在[不愿沉默如谜]()插入容祖儿的[重生](http://music.163.com/#/song?id=522631413)。点击页面中的“生成外链播放器”，将iframe代码放进原md文件中。但一开始有问题，iframe被当成普通的md文本。在[Jekyll raw HTML in post](https://stackoverflow.com/questions/30233461/jekyll-raw-html-in-post)中找到了答案。

网易云给的iframe代码为

```html
<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width=330 height=86 src="//music.163.com/outchain/player?type=2&id=522631413&auto=1&height=66"></iframe>
```

要将`width=330 height=86`改成`width="330" height="86"`，果然成功了。效果页面如下：

[![](music.png)](https://blog.hohoweiya.xyz/movie/2017/12/30/unwilling-to-be-silent.html)

## Jekyll add RSS feed

[RSS for Jekyll blogs](https://joelglovier.com/writing/rss-for-jekyll)

## jekyll tags 逗号分隔

采用

```jekyll
{% for tag in page.tags %}
    <a href="/tag/{{tag}}">{{tag}}</a>
    {% unless forloop.last %},{% endunless %}
{% endfor %}
```

但 [List of Dynamic Links in Jekyll](https://stackoverflow.com/questions/41858548/list-of-dynamic-links-in-jekyll) 提到了更完整的方案，

```jekyll
{% capture tagscommas %}
{% for tag in page.tags %}
    <a href="/tag/{{tag}}">{{tag}}</a>
    {% unless forloop.last %},{% endunless %}
{% endfor %}
{% endcapture %}

{{tagscommas}}
```

## Correct Jekyll

refer to [Configuring Jekyll for User and Project GitHub Pages](http://downtothewire.io/2015/08/15/configuring-jekyll-for-user-and-project-github-pages/)
