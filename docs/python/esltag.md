---
comments: true
---

# 自建关键词索引页

记录下为 ESL-CN 项目生成[关键词索引页](https://esl.hohoweiya.xyz/tag/index.html)的探索过程。

## 相关项目

第一步想找有没有实现类似功能的插件，因为想到 Jekyll 中也有那种设置标签页的，但是其实目标还不太一致，因为倘若按照 Jekyll 那样做，则需要把标签都放在 `metadata` 中，这样额外增加了工作量。

在此搜索相关项目的过程中，也发现了 (Material for ) Mkdocs 的一些此前未注意、未用到甚至未能理解的功能，即我刚刚说的 [metadata](https://squidfunk.github.io/mkdocs-material/extensions/metadata/)。

## 提取字串

既然没有，那就自己动手写代码实现。其实捋一捋需求挺简单的，就是想把每一节中的关键词单独拎出来放在一个页面中，而这些关键词是满足某个特定 pattern 的，于是自然而然用 `re`。虽然之前也零星用过几次 `re`，但是再次准备写的时候，还是要对着 reference 来摸索。

- 首先需要匹配中文，[这篇博客](https://blog.csdn.net/gatieme/article/details/43235791)给了很好的解答，采用 `([\u4e00-\u9fa5]+)`

- 其次匹配带有空格的英文词组，这个最后采用 `(\b[a-zA-Z ]+\b)`。这个尽量要精确，不然如果范围大点，企图依赖其他部分的 pattern 来限制住，那还是会出问题的，比如一开始用 `(\b.*\b)`，过大的匹配范围，最后得到些错误的匹配结果，其中 `-b` 匹配 word boundary.

读写部分，直接采用 

```python
file = open(filename, "rt")
contents = file.read()
file.close()
```

而列出文件夹中的所有文件采用 `os.listdir()`，而如果想找出特定的 pattern，比如我的需求是要求文件名前两位为章节 id，则需要适当判断，

```python
for idx, x in enumerate(docsdir):
    if f'{i:02}' in x:
        break
```

这里注意学习 `enumerate` 的用法，这应该还是很常见的，之前也用过，但没怎么记住。另外 `f{i:02}` 补零的技巧也要记住，其它实现方法参见 [How to pad zeroes to a string?](https://stackoverflow.com/questions/339007/how-to-pad-zeroes-to-a-string).

如果限定文件类型，有更方便的命令 `glob.glob()`，比如我只需要 `.md` 文件，则直接

```python
for file in glob.glob(f"docs/{chdir}/*.md"):
```

其他用法详见 [How do I list all files of a directory?](https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory)。

注意学会使用 `re` 的分组匹配功能，避免二次分割字串提取想要的元素。一开始我没用分组匹配，虽然得到了字串，但是后面想分割时，而 `split` 函数只支持一个 delimiter，而我想用多个 delimiter，比如对于 `**中文 (chinese)**`，为了提取出 `中文` 和 `chinese`，我需要 `**`、`(`、`)` 以及空格这四个 delimiter，虽然可以用

```python
str = "**中文 (chinese)**"
delimiters = "**", "(", ")", " "
regexPattern = '|'.join(map(re.escape, delimiters))

>>> regexPattern
'\\*\\*|\\(|\\)|\\ '
>>> re.split(regexPattern, str)
['', '\xe4\xb8\xad\xe6\x96\x87', '', 'chinese', '', '']
```

实现，但既然又用了 `re`，干嘛不直接用它的分组匹配功能？参考 [Python regex split without empty string](https://stackoverflow.com/questions/16840851/python-regex-split-without-empty-string) 及 [Split string with multiple delimiters in Python](https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python)

## 写入文件

想按照首字母分开，即 `A`, `B` 等作为小节标题，得到英文字母的技巧是 

```python
[chr(i+ord('A')) for i in range(26)]
```

其中 `ord` 和 `chr` 很实用。一开始的策略是准备长度为 26 的 list，每个 list 放入关键词的信息，但是这样会有重复，避免同一篇文章中的重复可以用 `list(set(some_list))`，但是这种也解决不了不同文件存在相同关键词的重复。后来采用 dict 存储后，可以把重复的文章链接添加到对应元素中去，但是这样对 dict 按照

```python
for k, v in d:
```

就会出问题，报错 

> not enough values to unpack (expected 2, got 1 

换一种遍历方式

```python
for key in d.keys():
```

详见 [not enough values to unpack (expected 2, got 1](https://stackoverflow.com/questions/52108914/not-enough-values-to-unpack-expected-2-got-1)

在用 `writelines` 出现很奇怪的现象，并没有换行，我猜会不会是系统的原因，即 `LF` 和 `CRLF` 的区别，但没细究，也有人在问这样的问题，后来直接采用

```python
tagpage.writelines(tag + '\n' for tag in tagi)
```

最后，我的代码为 [gentag.py@ESL-CN](https://github.com/szcf-weiya/ESL-CN/blob/master/gentag.py)。目前的版本只有首字母排序，没有进一步让后面字母排序，而是采用了默认读取的顺序。转念一想，这样也挺合适的。不过，或许某天我想改改呢，也说不定。