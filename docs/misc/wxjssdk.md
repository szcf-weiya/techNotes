# 微信自定义分享链接的内容（设置图片）

折腾历程 2018.3.13-2018.3.14

## 后端为 php

服务器后端通过 php 获取 access_token, ticket, 并计算签名，然后网页通过 ajax 访问获取 wx.config 的信息，经测试能够大致实现，但是当在又拍云中申请完 https 证书，通过 https 访问服务器，总是报出如下的错误信息

```none
{"msg":"connection refused","code":"50302004","id":"6ace947be627cf38ecd5065e36f9f25f"}
```

这样不能用 https 的话，在博客页面上访问 http 的资源不够清真。

## 后端为 flask

一开始其实是准备用 flask，但考虑到端口转发没弄，后来才发现端口转发其实很简单，php 本身不也是端口转发么？用 python 还是比较清真的。但似乎 https 仍存在问题，我现在怀疑很可能是当时有次配置错了，导致持续反应不过来，因为偶尔是正常的。搞定了基本代码

## 阿里云的免费证书好难找

虽然之前看到有说阿里云的免费证书，但一直在证书页面没有显示免费的选项，索性作罢，这还是我转而选择又拍云的一个因素。但今天奇迹地发现免费证书选项一直在那里，需要多点几下，才能切换处理。我的天哪！这样的话，我直接为服务器申请 CA 证书，不需要通过 CDN 中转，挺顺利的！

## 成功了吗

现在好像什么都准备好了，然而用 web 微信调试工具，分享时一直在报 error，但是提示信息确实分享成功，重复试验好多次，都是这个结果，都要崩溃了。索性不弄了，去跟室友打羽毛球了。

## 一场空

洗完澡发现，原来我压根就没有分享的权限，在公众号的开发权限那儿查看。哎，这就是命啊！

（完）

## 参考链接

1. [手把手带你使用JS-SDK自定义微信分享效果 - 归零back - 博客园](https://www.cnblogs.com/backtozero/p/7064247.html)
2. [微信JS SDK接入的几点注意事项 - 集君 - 博客园](https://www.cnblogs.com/chq3272991/archive/2017/06/22/7066614.html)
3. [Nginx 实现端口转发 - 星河赵 - 博客园](https://www.cnblogs.com/zhaoyingjie/p/7248678.html)