# CSS 笔记

CSS 全称为 Cascading Style Sheets, 中文名为**层叠样式表**.

??? note "元素"
	- 块级元素 (block-level element)
	- 内联级元素 (inline-level element)

??? note "选择器"
	- 类选择器：`.`
	- ID选择器: `#`
	- 属性选择器: `[]`
	- 伪类选择器: `:`
	- 伪元素选择器: `::`
	- 关系选择器
		- 后代: `<space>`
		- 相邻后代: `>`
		- 兄弟: `~`
		- 相邻兄弟: `+`

??? note "盒子"

	- 内在盒子:
		- content box: `content-box`
		- padding box: `padding-box`
		- border box: `border-box`
		- margin box: NONE

??? tip "`!important` not work in inline style"
	在 `DF` 项目中使用 `Documenter.jl` 时，因为画图保存格式为 pdf，所以想以 embed 的形式展示 pdf，但是发现高度很低，尝试过

	1. embed 外层再套一层 `div` 并设置高度
	2. 只用 embed 层，改变 `height` 为 `2100px`，`100%`

	通通不管用，然后发现是因为默认 embed 的 height 设成了 auto.

	一个自然想法是在 2100px 后面加 `!important`，但并没有成功，后来发现 [:link:](https://stackoverflow.com/questions/4616964/css-important-not-working)，所以需要单独写 css

	```css
	<style>
	embed {
		height: 2100px !important;
	}
	</style>
	```

## 博客中的定理环境

通过 css 实现，例如

```css
.theorem {
    display: block;
    margin: 12px 0;
    font-style: italic;
}
.theorem:before {
    content: "Theorem.";
    font-weight: bold;
    font-style: normal;
}
```

详见 [LaTeX Theorem-like Environments for the Web](http://drz.ac/2013/01/17/latex-theorem-like-environments-for-the-web/)

这种方法不能使用 markdown 的列表环境，有时候会不太方便。注意到 kramdown 本身具有一些特性可以解决这个问题，比如设置 Block Attributes，详见 [Quick Reference of kramdown](https://kramdown.gettalong.org/quickref.html)

则我可以用

>
{: .theorem}

实现定理环境，而且这样还有额外的好处，可以突出定理。但是 before 的字 "Theorem" 会单独占据一行，误打误撞看到 [Adding quotes to blockquote](https://stackoverflow.com/questions/32909991/adding-quotes-to-blockquote)，试了一下

```css
blockquote p {
    display: inline;
  }
```

可以解决这个问题，但担心会破坏其他的 blockquote 环境，于是指定 theorem 可以这样处理，即

```css
blockquote.theorem p {
    display: inline;
  }
```

这个用法参考 [CSS Id 和 Class](https://www.runoob.com/css/css-id-class.html)

## `width/height` 作用的具体细节

### `width:auto`

### `width:Xpx`

- `width:100px` 如何作用到 `<div>` 元素上？
- content box 环绕着 `width` 和 `height` 给定的矩形。由于 `<div>` 元素默认的 `padding`, `border` 和 `margin` 均为 0，则 `<div>` 呈现的像素也为 100px。

### `box-sizing`

用于改变 `width` 作用的盒子

![](https://user-images.githubusercontent.com/13688320/114684396-74a0d880-9d43-11eb-8fcf-4f0c3b135d8f.png)

### `height:100%`

如果包含块的高度没有显式指定（即高度由内容决定），并且该元素不是绝对定位，则计算值为 `auto`，而它和百分比是计算不了的，因此无效; 但如果包含块的宽度取决于该元素的宽度，产生的布局在 CSS2.1 中是未定义的，而浏览器可以自行处理未定义的行为。

=== "NOT WORK"
	```css
	div {
		width: 100%;
		height: 100%; /* invalid */
	}
	```

=== "NOT WORK"
	```css
	body {
		height: 100%;
	}
	```

=== "WORK"
	```css
	html, body {
		height: 100%;
	}
	```

=== "WORK"
	```css
	div {
		height: 100%;
		position: absolute;
	}
	```

## Chrome 固定 `:hover` 状态

在试图解决 [fix a problematic url #955](https://github.com/cosname/cosx.org/pull/955) 时，想通过 Chrome 调试看看 `:hover` 的具体 style 定义

![Peek 2021-08-27 11-34](https://user-images.githubusercontent.com/13688320/131088569-2329b315-6c39-42bc-8b5b-fea928482e8a.gif)

## em 和 rem

[何时使用 Em 与 Rem](https://www.w3cplus.com/css/when-to-use-em-vs-rem.html)

或者参考

[rem、px、em之间的区别以及网页响应式设计写法 - Gabriel_wei - 博客园](https://www.cnblogs.com/Gabriel-Wei/p/6180554.html)

简单说，

- `rem`: 相对根节点
- `em`: 相对父节点
- `px`: 绝对值


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
- [negative margins](https://stackoverflow.com/questions/11495200/how-do-negative-margins-in-css-work-and-why-is-margin-top-5-margin-bottom5)
