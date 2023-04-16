# Markdown

- [a simple example generated page via rmarkdown](https://bookdown.org/yihui/rmarkdown/rmarkdown-site.html#a-simple-example), see also [ESL-CN Issue 242](https://github.com/szcf-weiya/ESL-CN/issues/242)

## Typora

Personal Tips:

- copy image and then paste it into the editor, the image will also be copied into the cache folder of typora. To avoid that, just directly use the absolute path of the image. First copy the image file, and then paste it into the address bar in Chrome, which will display the absolute path like, `file://`.

## Collapsible Section

```bash
<details>
  <summary>Click to expand!</summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>
```

**an empty line after `</summary>`**

refer to [pierrejoubert73/markdown-details-collapsible.md](https://gist.github.com/pierrejoubert73/902cc94d79424356a8d20be2b382e1ab)


!!! tip "shortcut"
	It is annoying to input the tag manually. Here are I found two shortcuts. ([:link:](https://gist.github.com/pierrejoubert73/902cc94d79424356a8d20be2b382e1ab?permalink_comment_id=4201513#gistcomment-4201513))

	- bookmarklet ([:link:](https://www.freecodecamp.org/news/what-are-bookmarklets/)): add a javascript snippet as a bookmark.
	- [Text Blaze](https://chrome.google.com/webstore/detail/text-blaze/idgadaccgipmpannjkmfddolnnhmeklj/related): a chrome extension. Just type the customized define command `/details`.
  - **GitHub has enabled this feature.** [Slash Commands](https://github.com/orgs/community/discussions/40299)


## `mermaid`: flowchart

Homepage: <https://mermaid-js.github.io/>

!!! info "Private Examples"
    - [2021-06-23.md](https://github.com/szcf-weiya/SZmedinfo/blob/master/notes/2021-06-23.md)

- add newline in the text, use `<br />`. [:link:](https://github.com/mermaid-js/mermaid/issues/384)