## 利用@media screen实现网页布局的自适应

参考http://www.cnblogs.com/xcxc/p/4531846.html

### 1280分辨率以上（大于1200px）

```
@media screen and (min-width:1200px){
}
```

### 1100分辨率（大于960px，小于1199px）
```
@media screen and (min-width: 960px) and (max-width: 1199px) {

}
```

### 880分辨率（大于768px，小于959px）

```
@media screen and (min-width: 768px) and (max-width: 959px) {

}
```
## display

对img添加
```
display: block
```
达到居中的作用

## meta

<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
```
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
```

1. http://www.w3cplus.com/html5/meta-tags-and-seo.html
2. http://www.cnblogs.com/st-leslie/p/5196852.html

## font-family的选择

看到一篇超级棒的博客

[如何优雅的选择字体(font-family)](https://segmentfault.com/a/1190000006110417)

## cloudflare 的rocket load module
慎用，在ESL上，出现了`A Parser-blocking, cross-origin script, http://example.org/script.js, is invoked via document.write. This may be blocked by the browser if the device has poor network connectivity.` bug，以及公式解析错误。

解决方案，去cloudflare主页disable掉rocket load module.

## Uncaught ReferenceError: $ is not defined?

参考[Uncaught ReferenceError: $ is not defined?](https://stackoverflow.com/questions/2075337/uncaught-referenceerror-is-not-defined?page=1&tab=votes#tab-top)

## 网页底部出现滚动条

查看footer的width是否为100%，若是，则删掉，某次是因为这个原因。

## footer 保持在底部

[告诉你一个将 footer 保持在底部的最好方法](https://www.jianshu.com/p/4896e6936ce3)

## em 和 rem

[何时使用 Em 与 Rem](https://www.w3cplus.com/css/when-to-use-em-vs-rem.html)

或者参考

[rem、px、em之间的区别以及网页响应式设计写法 - Gabriel_wei - 博客园](https://www.cnblogs.com/Gabriel-Wei/p/6180554.html)
