## Github Pages 与百度爬虫

[解决 Github Pages 禁止百度爬虫的方法与可行性分析](http://jerryzou.com/posts/feasibility-of-allowing-baiduSpider-for-Github-Pages/)

[利用 CDN 解决百度爬虫被 Github Pages 拒绝的问题](https://www.dozer.cc/2015/06/github-pages-and-cdn.html)

### 索引量减少到几乎为 0 了

![](baidu-index.png)

然后发现原来是忘记在百度专用服务器上 `git checkout gh-pages`了，导致直接访问失败，`git log` 显示，上一次更新时间还是 20190804，时间点差不多能对得上去。

### HTTPS 认证

为了让百度收录，所以单独为百度设置了域名解析，但是这种情况下不是 https，不能像正常情况下通过 CDN 申请 https。注意到阿里云可以申购免费证书，并下载，而且提供不同下载需求，比如针对 nginx 或 apache 的。直接按照帮助文档走一遍就好了。
