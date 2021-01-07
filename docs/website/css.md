# CSS 常见用法

## 自适应网页布局

参考 [利用@media screen实现网页布局的自适应](http://www.cnblogs.com/xcxc/p/4531846.html)

### 1280分辨率以上（大于1200px）

```css
@media screen and (min-width:1200px){
}
```

### 1100分辨率（大于960px，小于1199px）

```css
@media screen and (min-width: 960px) and (max-width: 1199px) {
}
```

### 880分辨率（大于768px，小于959px）

```css
@media screen and (min-width: 768px) and (max-width: 959px) {
}
```

## display

对 img 添加

```
display: block
```

达到居中的作用

## meta 标签

> Meta标签给搜索引擎提供了许多关于网页的信息。

为了不让用户手动去改变页面大小，需要加上

```html
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
```

1. [Meta 标签与搜索引擎优化](http://www.w3cplus.com/html5/meta-tags-and-seo.html)
2. [手机网页布局经验总结](http://www.cnblogs.com/st-leslie/p/5196852.html)

## font-family的选择

看到一篇超级棒的博客

[如何优雅的选择字体(font-family)](https://segmentfault.com/a/1190000006110417)

## cloudflare 的rocket load module

慎用，在ESL上，出现了如下问题

> A Parser-blocking, cross-origin script, http://example.org/script.js, is invoked via document.write. This may be blocked by the browser if the device has poor network connectivity.

以及公式解析错误。

解决方案，去cloudflare主页disable掉rocket load module.

## Uncaught ReferenceError: $ is not defined?

参考[Uncaught ReferenceError: $ is not defined?](https://stackoverflow.com/questions/2075337/uncaught-referenceerror-is-not-defined?page=1&tab=votes#tab-top)

## 网页底部出现滚动条

查看footer的width是否为100%，若是，则删掉，某次是因为这个原因。

## footer 保持在底部

通过 `flexbox`，例如

```css
/* https://demo.tutorialzine.com/2016/03/quick-tip-the-best-way-to-make-sticky-footers/styles.css */
body{
    display: flex;
    flex-direction: column;
    height: 100%;
	font: 400 15px/1.4 'Open Sans',sans-serif;
}

header{
	/* We want the header to have a static height - it will always take up just as much space as it needs.  */
	/* 0 flex-grow, 0 flex-shrink, auto flex-basis */
	flex: 0 0 auto;
}

.main-content{
	/* By setting flex-grow to 1, the main content will take up
	all of the remaining space on the page (the other elements have flex-grow: 0 and won't contest the free space). */
	/* 1 flex-grow, 0 flex-shrink, auto flex-basis */
	flex: 1 0 auto;
}

footer{
	/* Just like the header, the footer will have a static height - it shouldn't grow or shrink.  */
	/* 0 flex-grow, 0 flex-shrink, auto flex-basis */
	flex: 0 0 auto;
}
```

参考

- [告诉你一个将 footer 保持在底部的最好方法](https://www.jianshu.com/p/4896e6936ce3)
- [Quick Tip: The Best Way To Make Sticky Footers](https://demo.tutorialzine.com/2016/03/quick-tip-the-best-way-to-make-sticky-footers/)

## em 和 rem

[何时使用 Em 与 Rem](https://www.w3cplus.com/css/when-to-use-em-vs-rem.html)

或者参考

[rem、px、em之间的区别以及网页响应式设计写法 - Gabriel_wei - 博客园](https://www.cnblogs.com/Gabriel-Wei/p/6180554.html)

简单说，

- `rem`: 相对根节点
- `em`: 相对父节点
- `px`: 绝对值

## Mathjax 公式锚点偏移

详见 [mathjax link to equation](https://github.com/szcf-weiya/ESL-CN/issues/173)

参考资料：

- [`:target` 选择器](https://www.runoob.com/cssref/sel-target.html): 用于锚`#`元素的样式
- [锚点定位向下偏移](https://www.cnblogs.com/shenjp/p/11088344.html)
- scroll issue mentioned in mkdocs-material:
  - [scroll issue when link anchor is in a table header #746](https://github.com/squidfunk/mkdocs-material/issues/746)
  - [and the changes seems only `scroll-margin-top`](https://github.com/squidfunk/mkdocs-material/blob/9a0c3e9094256a41d695da00afca733201406f43/src/assets/stylesheets/extensions/_permalinks.scss)
- [css中一个冒号和两个冒号](https://zhuanlan.zhihu.com/p/161187023)
- [css 伪元素](https://www.runoob.com/css/css-pseudo-elements.html)
- [offsets anchor to adjust for fixed header](https://stackoverflow.com/questions/10732690/offsetting-an-html-anchor-to-adjust-for-fixed-header)
- [Fixed page header overlaps in-page anchors](https://stackoverflow.com/questions/4086107/fixed-page-header-overlaps-in-page-anchors)
