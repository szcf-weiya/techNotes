# Python Notes

## Base

### xrange vs range

基本都在循环时使用，输出结果也是一样的，但略有差异

- range 直接生成一个 list 对象
- xrange 返回一个生成器。性能会比 range 好，特别是很大的时候。

参考[Python中range和xrange的区别](http://blog.csdn.net/imzoer/article/details/8742283)

### `-m`

`python -m` lets you run modules as scripts, and it reflects the motto--"batteries included". [Here](http://pythonwise.blogspot.com/2015/01/python-m.html) are some powerful features/functions, such as creating a simple http server

```python
python -m SimpleHTTPServer 80
```

### with 语句

参考

- [Python with语句](https://www.cnblogs.com/zhangkaikai/p/6669750.html)
- [with statement in Python](https://www.geeksforgeeks.org/with-statement-in-python/)

简言之，“使用with后不管with中的代码出现什么错误，都会进行对当前对象进行清理工作。”

这也就是为什么在用 MySQLdb 的时候，称“With the with keyword, the Python interpreter automatically releases the resources. It also provides error handling.” 详见[MySQL Python tutorial - programming MySQL in Python](http://zetcode.com/db/mysqlpython/)

另外，现在经常在神经网络中碰到，如

```python
import tensorflow as tf
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2    
dy_dx = tape.gradient(y, x)
```

### usage of `yield`

> We should use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory.

More details refer to [When to use yield instead of return in Python?](https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/)

An example gave in [Squaring all elements in a list](https://stackoverflow.com/questions/12555443/squaring-all-elements-in-a-list)

```python
>>> def square(list):
...     for i in list:
...             yield i ** 2
... 
>>> square([1,2])
<generator object square at 0x7f343963dca8>
>>> for i in square([1,2]):
...     print(i)
... 
1
4
```

along with other methods to square a list,

```python
>>> list = [1, 2]
>>> [i**2 for i in list]
[1, 4]
>>> map(lambda x: x**2, list)
<map object at 0x7f3439642f60>
>>> for i in map(lambda x: x**2, list):
...     print(i)
... 
1
4
>>> ret = []
>>> for i in list:
...     ret.append(i**2)
... 
>>> ret
[1, 4]
```

## Class

### 新式类 vs 经典类 (`class ClassName` vs `class ClassName(object)`)

- python 2.x 中，默认为经典类，只有当写成 `class A(object)` 才成为新式类
- python 3.x 中，默认为新式类

详见 [python新式类和经典类的区别？](https://www.zhihu.com/question/22475395)

1. In python 2.x, when you inherit from "object" you class is a "new style" class; the non inheriting from "object" case creates an "old style" class.
2. In python 3.x, all classes are new style - no need to set the metaclass.

refer to [class ClassName versus class ClassName(object)](https://stackoverflow.com/questions/10043963/class-classname-versus-class-classnameobject)

### `__getitem__` and `__setitem__`

I came across the usage of `__getitem__` [here](https://github.com/MeepMoop/tilecoding/blob/master/example.py#L34), which seems powerful, and not only accept

```python
T[x, y]
```

also supports

```python
T[[x, y]]
```

in [my code](https://github.com/szcf-weiya/RLnotes/blob/8e714286c1ba09113c4bf295d89ed774a8c5be5c/ModelFree/mountaincar.py#L68), where `T` is an instance of a class and `[x, y]` is the coordinate. The `[]` is enabled due to the `__getitem__` method.

```python
class TildCoder():
	def __init(...):
		...
	
	def __getitem__(self, x):
		...
```

Then I found [more detailed explanation](https://stackoverflow.com/a/43627975/) for the usage.

> The `[]` syntax for getting item by key or index is just syntax sugar. When you evaluate `a[i]`, Python calls `a.__getitem__(i)` or `type(a).__getitem__(a, i)`.

### @staticmethod vs @classmethod

参考

1. [Difference between @staticmethod and @classmethod in Python](https://www.pythoncentral.io/difference-between-staticmethod-and-classmethod-in-python/)

2. [The definitive guide on how to use static, class or abstract methods in Python](https://julien.danjou.info/guide-python-static-class-abstract-methods/)

## CLI Options

### `sys.argv`

The arguments are stored in the array `sys.argv`, where the first element is the script name itself.

```python
--8<-- "docs/python/ex/star_arg.py"
```

With quote, the character `*` would expanding before passing into the argument.

```bash
$ python star_arg.py ../*.png
['star_arg.py', '../error_jupyter_1.png', '../error_jupyter.png', '../err_theano.png', '../ipython.png', '../spyder-qtconsole.png']
$ python star_arg.py "../*.png"
['star_arg.py', '../*.png']
```

If we want to wrap it with a bash function,

```bash
$ star_arg() { python star_arg.py $1 ; }
```

The quote for the pattern containing `*` is also necessary, otherwise it first expands and just passes the first matched filename due to `$1`,

```bash
$ star_arg ../*.png
['star_arg.py', '../error_jupyter_1.png']
$ star_arg "../*.png"
['star_arg.py', '../error_jupyter_1.png', '../error_jupyter.png', '../err_theano.png', '../ipython.png', '../spyder-qtconsole.png']
```

But note that the above bash function does not wrap `$1`. With quote, the original string can be passed into python script

```bash
$ star_arg2() { python star_arg.py "$1" ; }
$ star_arg2 "../*.png"
['star_arg.py', '../*.png']
$ star_arg2 ../*.png
['star_arg.py', '../error_jupyter_1.png']
```


### `argparse`

Documentation: [argparse — Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html)

An example: [jieba/__main__.py](https://github.com/fxsjy/jieba/blob/master/jieba/__main__.py)

### `click`

!!! info
	More related code can be found in [My Code Results on GitHub](https://github.com/search?l=Python&q=user%3Aszcf-weiya+click&type=Code)

It enables to use command line arguments. For example, 

```python
--8<-- "docs/python/ex/ex_click.py"
```

```bash
$ python3 ex_click.py --a 1 --b
a = 1, b = True
$ python3 ex_click.py --a 1
a = 1, b = False
$ python3 ex_click.py --a 2
a = 2, b = False
```

!!! warning
	The script filename cannot be the same as the module name, `click.py`. Otherwise, it throws,
	```bash
	$ python3 click.py --a 1 --b
	Traceback (most recent call last):
	File "click.py", line 1, in <module>
		import click
	File "/media/weiya/Seagate/GitHub/techNotes/docs/python/ex/click.py", line 3, in <module>
		@click.command()
	AttributeError: module 'click' has no attribute 'command'
	```

!!! failure "float value"
	With 
	```python
	@click.option("--c", default = 2)
	```
	it throws
	```bash
	$ python ex_click.py --a 1 --b --c 2.0
	Usage: ex_click.py [OPTIONS]
	Try 'ex_click.py --help' for help.

	Error: Invalid value for '--c': 2.0 is not a valid integer
	```

	`type = float` needs to be specified.

## FileIO

### read two files simultaneously

```python
with open("file1.txt") as f1, open("file2.txt) as f2:
    for l1, l2 in zip(f1, f2):
        l1.strip() # rm `\n`
        l2.strip()
```

refer to [Reading two text files line by line simultaneously - Stack Overflow](https://stackoverflow.com/questions/11295171/reading-two-text-files-line-by-line-simultaneously)

!!! info
    [A practical example](https://github.com/szcf-weiya/SZmedinfo/blob/f8ea37c88c862affb3430e9a1278b9a261c80e83/src/trajectory.py#L10)

### 写入 non-ascii 字符

```python
f = open("filename", "w")
write_str = u'''
some non ascii symbols
'''
f.write(write.str)
```

会报错

```
'ascii' codec can't encode character
```

参考 [Python: write a list with non-ASCII characters to a text file](https://stackoverflow.com/questions/33255846/python-write-a-list-with-non-ascii-characters-to-a-text-file) 采用 `codecs.open(, "w", encoding="utf-8")` 可以解决需求。

### URL string to normal string

参考[transform-url-string-into-normal-string-in-python-20-to-space-etc](https://stackoverflow.com/questions/11768070/transform-url-string-into-normal-string-in-python-20-to-space-etc)

1. python2

```python
import urllib2
print urllib2.unquote("%CE%B1%CE%BB%20")
```

2. python3

```python
from urllib.parse import unquote
print(unquote("%CE%B1%CE%BB%20"))
```

### `<U5`

I met this term in [How to convert numpy object array into str/unicode array?](https://stackoverflow.com/questions/16037824/how-to-convert-numpy-object-array-into-str-unicode-array)

I am confused about the [official english documentation](https://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.dtypes.html)

[A Chinese answer](https://segmentfault.com/q/1010000012049371) solves my question,

- `<` 表示字节顺序，小端（最小有效字节存储在最小地址中）
- `U` 表示Unicode，数据类型
- `5` 表示元素位长，数据大小

### `u/U`, `r/R`, `b` in string

- u/U: 表示unicode字符串。不是仅仅是针对中文, 可以针对任何的字符串，代表是对字符串进行unicode编码。一般英文字符在使用各种编码下, 基本都可以正常解析, 所以一般不带u；但是中文, 必须表明所需编码, 否则一旦编码转换就会出现乱码。建议所有编码方式采用utf8

- r/R: 非转义的原始字符串，常用于正则表达式 re 中。

- b:bytes:
       - python3.x里默认的str是(py2.x里的)unicode, bytes是(py2.x)的str, b”“前缀代表的就是bytes
       - python2.x里, b前缀没什么具体意义， 只是为了兼容python3.x的这种写法

另外

- `str` -> `bytes`: encode
- `bytes` -> `str`: decode 

```python
# python 3.6: str 为 Unicode
>>> "中文".encode("utf8")
b'\xe4\xb8\xad\xe6\x96\x87'
>>> "中文".encode("utf8").decode("utf8")
'中文'

# python 2.7： str 为 bytes
>>> "中文"
'\xe4\xb8\xad\xe6\x96\x87'
>>> "中文".decode("utf8")
u'\u4e2d\u6587'
>>> print("中文".decode("utf8"))
中文
```

参考:

- [python字符串前面加u,r,b的含义](https://www.oschina.net/question/437227_106832)
- [浅析Python3中的bytes和str类型 - Chown-Jane-Y - 博客园](https://www.cnblogs.com/chownjy/p/6625299.html)
- [convert-bytes-to-a-string](https://stackoverflow.com/questions/606191/convert-bytes-to-a-string)

## Function 

### `*args` and `**args`

- `*args`: pass a **non-keyword** and **variable-length** argument list to a function.
- `**args`: pass a **keyworded**, **variable-length** argument list, actually `dict`

refer to [`*args` and `**kwargs` in Python](https://www.geeksforgeeks.org/args-kwargs-python/), and [Asterisks in Python: what they are and how to use them](https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/)

One example,

```python
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)
```

where `*self.input_shape` aims to unpacking something like `[3, 4, 5]` to `3, 4, 5`.

### annotations

When I am writing the custom loss function in XGBoost, there are some new syntax in the example function,

```python
def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
```

what is the meaning of `:` and `->`. Then I [found](https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions) that they are [functional annotations](https://www.python.org/dev/peps/pep-3107/).

> By itself, Python does not attach any particular meaning or significance to annotations. 
>
> The only way that annotations take on meaning is when they are interpreted by third-party libraries. 

### No `Tuple` here

Actually, seems no need to add `Tuple` before `[np.ndarray, np.ndarray]`, which will throws an error 

> NameError: name 'Tuple' is not defined

Ohhh, to avoid such problem, 

```python
from typing import Tuple, Dict, List
```

refer to [custom_rmsle.py#L16](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_rmsle.py#L16)

## IPython

### Built-in magic commands 

Refer to [:link:](https://ipython.readthedocs.io/en/stable/interactive/magics.html) for more details.

- `%matplotlib inline` enables the inline backend for usage with the IPython notebook.
- `!ls` simply calls system's `ls`, but `!!ls` also returns the result formatted as a list, which is equivalent to `%sx ls`.
- `%sx`: shell execute, run shell command and capture output, and `!!` is short-hand.

### remote ipython kernel

一直想玩 jupyter 的远程 ipython kernel 连接，这样可以减轻本机的压力。

这两篇介绍得很详细，但是最后设置 ssh 那一步总是报错，总是说无法连接。

- [Connecting Spyder IDE to a remote IPython kernel](https://medium.com/@halmubarak/connecting-spyder-ide-to-a-remote-ipython-kernel-25a322f2b2be)
- [How to connect your Spyder IDE to an external ipython kernel with SSH PuTTY tunnel](https://medium.com/@mazzine.r/how-to-connect-your-spyder-ide-to-an-external-ipython-kernel-with-ssh-putty-tunnel-e1c679e44154)

因为我是直接把 `id_rsa.pub` 文件当作 `.pem` 文件，但如果我换成密码登录后就成功了。

而如果[直接命令行操作](https://github.com/ipython/ipython/wiki/Cookbook:-Connecting-to-a-remote-kernel-via-ssh)，则就像正常 ssh 一样，也会成功。

所以中间的差异应该就是 `.pem` 与 `id_rsa.pub` 不等同。具体比较详见 [what is the difference between various keys in public key encryption](https://stackoverflow.com/questions/17670446/what-is-the-difference-between-various-keys-in-public-key-encryption)


## JSON

### convert string to json

```python
payload='{'name': weiya}'
# payload='payload = {'name': weiya}'
```

换成json

```python
json.loads(payload)
```

!!! warning
    注意不能采用注释掉的部分。

### json.dumps() 和 json.dump() 的区别

简言之，`dumps()`和`loads()`都是针对字符串而言的，而`dump()`和`load()`是针对文件而言的。具体细节参见[python json.dumps()  json.dump()的区别 - wswang - 博客园](https://www.cnblogs.com/wswang/p/5411826.html)

### flask 中 jsonify 和 json.dumps 的区别

参考[在flask中使用jsonify和json.dumps的区别](http://blog.csdn.net/Duke_Huan_of_Qi/article/details/76064225)

另外 flask 的入门文档见

[快速入门 &mdash; Flask 0.10.1 文档](http://docs.jinkan.org/docs/flask/quickstart.html#quickstart)

## Jupyter

### GitHub 语言比例过分倾斜

[LInguist is reporting my project as a Jupyter Notebook](https://github.com/github/linguist/issues/3316)

### jupyter notebook 出错

![](error_jupyter.png)

可以通过
```
rm -r .pki
```
解决


### 创建 jupyter notebook 权限问题

![](error_jupyter_1.png)

原因是所给的路径的用户权限不一致，jupyter的用户及用户组均为root，为解决这个问题，直接更改用户权限

```
sudo chown weiya jupyter/ -R
sudo chgrp weiya jupyter/ -R
```
其中-R表示递归调用，使得文件夹中所有内容的用户权限都进行更改。

### `nbconvert failed: validate() got an unexpected keyword argument 'relax_add_props'`

refer to [nbconvert failed: validate() got an unexpected keyword argument 'relax_add_props' #2901](https://github.com/jupyter/notebook/issues/2901)

其实我的版本是一致的，但可能由于我进入 Jupyter notebook 方式不一样。

- 一开始，直接从base 进入，然后选择 snakes 的 kernel，导出失败，错误原因如上
- 直接在 snakes 进入 Jupyter notebook，这样可以成功导出

### different kernels

#### Python

其实不用对每个 environment 安装单独的 jupyter，只要安装 `ipykernel` 就好，这样都能从 base environment 中通过 jupyter 来选择不同 kernel，详见 [Kernels for different environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments)

```bash
$ conda activate myenv
$ conda install ipykernel
$ python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

#### Julia

打开特定版本的 Julia，

```julia
> add IJulia
```

#### R

```R
install.packages('IRkernel')
#IRkernel::installspec()
IRkernel::installspec(name="3.6.0", displayname = "R 3.6.0")
```

另见 [using R in JupyterLab](../R/index.md#using-r-in-jupyterlab)

#### management

列出所有安装的 kernel，

```bash
$ jupyter kernelspec list
Available kernels:
  3.6.0        /home/project09/.local/share/jupyter/kernels/3.6.0
  ir           /home/project09/.local/share/jupyter/kernels/ir
  julia-1.1    /home/project09/.local/share/jupyter/kernels/julia-1.1
  julia-1.4    /home/project09/.local/share/jupyter/kernels/julia-1.4
  mu-lux-cz    /home/project09/.local/share/jupyter/kernels/mu-lux-cz
  sam          /home/project09/.local/share/jupyter/kernels/sam
  python3      /home/project09/miniconda3/share/jupyter/kernels/python3
```

但是没有显示出 display name，其定义在文件夹下的 `kernel.json` 文件中（refer to [How to Change Jupyter Notebook Kernel Display Name](https://stackoverflow.com/questions/48960699/how-to-change-jupyter-notebook-kernel-display-name)）

```bash
$ jupyter kernelspec list | sed -n '2,$p' | awk '{print $2}' | xargs -I {} grep display {}/kernel.json
  "display_name": "R 3.6.0",
  "display_name": "R",
  "display_name": "Julia 1.1.1",
  "display_name": "Julia 1.4.2",
 "display_name": "py37",
 "display_name": "py37 (sam)",
 "display_name": "Python 3",
```

为了打印出对应的 env 名，

```bash
$ paste <(jupyter kernelspec list | sed -n '2,$p') <(jupyter kernelspec list | sed -n '2,$p' | awk '{print $2}' | xargs -I {} grep display {}/kernel.json)
  3.6.0        /home/project09/.local/share/jupyter/kernels/3.6.0	  "display_name": "R 3.6.0",
  ir           /home/project09/.local/share/jupyter/kernels/ir	  "display_name": "R",
  julia-1.1    /home/project09/.local/share/jupyter/kernels/julia-1.1	  "display_name": "Julia 1.1.1",
  julia-1.4    /home/project09/.local/share/jupyter/kernels/julia-1.4	  "display_name": "Julia 1.4.2",
  mu-lux-cz    /home/project09/.local/share/jupyter/kernels/mu-lux-cz	 "display_name": "py37",
  sam          /home/project09/.local/share/jupyter/kernels/sam	 "display_name": "py37 (sam)",
  python3      /home/project09/miniconda3/share/jupyter/kernels/python3	 "display_name": "Python 3",
```

!!! question
	此处很不优雅地重新复制了一下 `jupyter kernelspec list | sed -n '2,$p'`，在 pipeline 中是否有更直接的方法？

## List

### find index of an item

```python
>>> [1, 1].index(1)
0
>>> [i for i, e in enumerate([1, 2, 1]) if e == 1]
[0, 2]
>>> g = (i for i, e in enumerate([1, 2, 1]) if e == 1)
>>> next(g)
0
>>> next(g)
2
```

refer to [Finding the index of an item given a list containing it in Python](https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python)

### index a list with another list

```python
L = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
Idx = [0, 3, 7]
T = [L[i] for i in Idx]
```

refer to [In Python, how do I index a list with another list?](https://stackoverflow.com/questions/1012185/in-python-how-do-i-index-a-list-with-another-list)

### getting indices of true

```python
>>> t = [False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False]
>>> [i for i, x in enumerate(t) if x]
[4, 5, 7]
```

refer to [Getting indices of True values in a boolean list](https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list)

### remove by index

`del`

refer to [How to remove an element from a list by index?](https://stackoverflow.com/questions/627435/how-to-remove-an-element-from-a-list-by-index)

### `TypeError: unhashable type: 'list'`

#### convert a nested list to a list

```python
Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
[GCC 8.4.0] on linux
>>> set([1,2,3,4,[5,6,7],8,9])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
>>> set([1,2,3,4,(5,6,7),8,9])
{1, 2, 3, 4, (5, 6, 7), 8, 9}
```

#### hash a nested list

```python
>>> hash([1, 2, 3, [4, 5,], 6, 7])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
>>> hash(tuple([1, 2, 3, [4, 5,], 6, 7]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
>>> hash(tuple([1, 2, 3, tuple([4, 5,]), 6, 7]))
-7943504827826258506
>>> hash([1, 2, 3, tuple([4, 5,]), 6, 7])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
```

refer to [Python: TypeError: unhashable type: 'list'](https://stackoverflow.com/questions/13675296/python-typeerror-unhashable-type-list)

### convert list of lists to list

```bash
>>> a = [[1, 2], [3, 4]]
>>> [y for x in a for y in x]
[1, 2, 3, 4]
```

refer to [How to make a flat list out of a list of lists?](https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists)

## matplotlib.pyplot

### seaborn

Homepage: [seaborn: statistical data visualization](https://seaborn.pydata.org/)

> Seaborn的底层是基于Matplotlib的，他们的差异有点像在点餐时选套餐还是自己点的区别，Matplotlib是独立点菜，可能费时费心（尤其是对我这种选择困难症患者...）但最后上桌的菜全是特别适合自己的；而Seaborn是点套餐，特别简单，一切都是配好的，虽然省时省心，但可能套餐里总有些菜是不那么合自己口味的。
>
> :link: [Jack Sun @ 知乎](https://www.zhihu.com/question/301637122/answer/528183410)

### order of xy

`imshow` 中的 [origin and extent](https://matplotlib.org/tutorials/intermediate/imshow_extent.html)

> Generally, for an array of shape (M, N), the first index runs along the vertical, the second index runs along the horizontal. The pixel centers are at integer positions ranging from 0 to N' = N - 1 horizontally and from 0 to M' = M - 1 vertically. `origin` determines how to the data is filled in the bounding box.


an illustrative example,

```python
--8<-- "docs/python/plt/xy.py"
```

![image](https://user-images.githubusercontent.com/13688320/125582405-cf8a7af7-e341-44ed-a251-d867a8c4af2c.png)


which is adapted from [matplotlib: coordinates convention of image imshow incompatible with plot](https://stackoverflow.com/questions/37706005/matplotlib-coordinates-convention-of-image-imshow-incompatible-with-plot)



### subplots 的间距

`plt.tight_layout()` 可以调节间距，如果有必要，可以带上参数，比如，[B spline in R, C++ and Python](https://github.com/szcf-weiya/ESL-CN/commit/a79daf246320a7cd0ae57c0b229fc096d98483f6)

```bash
plt.tight_layout(pad = 3.0)
```

### equal axis aspect ratio

According to the [official documentation](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axis_equal_demo.html),

```python
fig, ax = plt.subplots(1, 3)
ax[0].plot(3*np.cos(an), 3*np.sin(an))
ax[1].plot(3*np.cos(an), 3*np.sin(an))
ax[1].axis("equal")
ax[2].plot(3*np.cos(an), 3*np.sin(an))
ax[2].set_aspect("equal", "box")
```

![image](https://user-images.githubusercontent.com/13688320/122642147-7cae3780-d13b-11eb-9e14-9356d2e2f6a9.png)

we prefer to the last one. If we want to directly call `plt` instead of `fig, ax`, then

```python
plt.plot(3*np.cos(an), 3*np.sin(an))
plt.gca().set_aspect("equal", "box")
```

note the middle `gca()`.

### scatter size

the size is defined by the area, [pyplot scatter plot marker size](https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size)

### interactive mode

- [How to pause a for loop and waiting for user input matplotlib](https://stackoverflow.com/questions/47273107/how-to-pause-a-for-loop-and-waiting-for-user-input-matplotlib)
- [Event handling and picking](https://matplotlib.org/3.1.1/users/event_handling.html)

!!! info
	Related projects:
	- [pause for stopping and resuming to plot](https://github.com/szcf-weiya/Cell-Video/blob/4721ef10b6f77f59dbed639c6806faa1b644ba06/models/BF_single.py#L166-L167)
	- [plot every X seconds](https://bitbucket.org/weiya/fruitcup/src/3880402ca0ef892eaca6856a93ee026676e80fab/src/fruitcup_solution_wlj/FruitCup.py#lines-347:348)

### math symbols

```python
plt.xlabel(r"\lambda")
```

refer to [Writing mathematical expressions](https://matplotlib.org/users/mathtext.html)

### Show matplotlib plots in Ubuntu (Windows subsystem for Linux)

参考
[Show matplotlib plots in Ubuntu (Windows subsystem for Linux)](https://stackoverflow.com/questions/43397162/show-matplotlib-plots-in-ubuntu-windows-subsystem-for-linux)


## NumPy

### `(R, 1)` vs `(R, )`

> The best way to think about NumPy arrays is that they consist of two parts, a **data buffer** which is just a block of raw elements, and a **view** which describes how to interpret the data buffer. [:material-stack-overflow:](https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r)

```python
>>> np.shape(np.ones((3, 1)))
(3, 1)
>>> np.shape(np.ones((3, )))
(3,)
>>> np.shape(np.ones((3)))
(3,)
>>> np.shape(np.ones(3))
(3,)
>>> np.shape(np.ones(3, 1)) # ERROR!
```

### array operation

```python
>>> a = np.sum([[0, 1.0], [0, 5.0]], axis=1)
>>> c = np.sum([[0, 1.0], [0, 5.0]], axis=1, keepdims=True)
>>> a/c
array([[ 1. ,  5. ],
       [ 0.2,  1. ]])
>>> a
array([ 1.,  5.])
>>> c
array([[ 1.],
       [ 5.]])
>>> d
array([[ 1,  5],
      [ 0, 10]])
>>> d/c
array([[ 1.,  5.],
      [ 0.,  2.]])
>>> d/a
array([[ 1.,  1.],
      [ 0.,  2.]])
```

### arrays with different size

- nested list

```python
[[1,2,3],[1,2]]
```

- numpy

```python
numpy.array([[0,1,2,3], [2,3,4]], dtype=object)
```

refer to [How to make a multidimension numpy array with a varying row size?](https://stackoverflow.com/questions/3386259/how-to-make-a-multidimension-numpy-array-with-a-varying-row-size)


### `np.newaxis`

add new dimensions to a numpy array [:material-stack-overflow:](https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array).

```python
>>> a = np.ones((2, 3))
>>> np.shape(a[:, :, None])
(2, 3, 1)
>>> np.shape(a[:, :, np.newaxis])
(2, 3, 1)
```

### array of FALSE/TRUE

```python
np.zeros(10, dtype = bool)
```

### Array of Array

```python
In [29]: a
Out[29]: 
array([[ 83.,  11.],
       [316.,  19.],
       [372.,  35.]])

In [30]: np.hstack([a, a])
Out[30]: 
array([[ 83.,  11.,  83.,  11.],
       [316.,  19., 316.,  19.],
       [372.,  35., 372.,  35.]])

In [31]: b = np.array([a, a])

In [32]: b[1]
Out[32]: 
array([[ 83.,  11.],
       [316.,  19.],
       [372.,  35.]])
```

### print numpy objects without line breaks

```python
import numpy as np
x_str = np.array_repr(x).replace('\n', '')
print(x_str)
```

refer to [How to print numpy objects without line breaks](https://stackoverflow.com/questions/29102955/how-to-print-numpy-objects-without-line-breaks)

### merge multiple slices

```python
x = np.empty((15, 2))
x[np.r_[0:5, 10:15],:]
```

refer to [Numpy: An efficient way to merge multiple slices](https://stackoverflow.com/questions/46640821/numpy-an-efficient-way-to-merge-multiple-slices)

but it does not allow the iterator to construct the list,

```ipython
In: [i:i+3 for i in 1:3]
  File "<ipython-input-247-8e21d57e8f2d>", line 1
    [i:i+3 for i in 1:3]
      ^
SyntaxError: invalid syntax

In: np.r_[i:i+3 for i in 1:3]
  File "<ipython-input-246-02f2b1ded12b>", line 1
    np.r_[i:i+3 for i in 1:3]
                ^
SyntaxError: invalid syntax
```

As a comparison, Julia seems more flexible,

```julia
julia> vcat([i:i+3 for i in 1:3]...)'
1×12 adjoint(::Vector{Int64}) with eltype Int64:
 1  2  3  4  2  3  4  5  3  4  5  6
```

### `/=`

According to the builtin manual, `help('/=')`,

> An augmented assignment expression like "x += 1" can be rewritten as
"x = x + 1" to achieve a similar, but not exactly equal effect. In the
augmented version, "x" is only evaluated once. Also, when possible,
the actual operation is performed *in-place*, meaning that rather than
creating a new object and assigning that to the target, the old object
is modified instead.

So it would be cautious when using it on arrays,

```python
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> a1 = a[0, ]
>>> a1 /= 0.5
>>> a1
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-106-71f04990b2bb> in <module>
      1 a = np.array([[1, 2, 3], [4, 5, 6]])
      2 a1 = a[0, ]
----> 3 a1 /= 0.5
      4 a1

TypeError: ufunc 'true_divide' output (typecode 'd') could not be coerced to provided output parameter (typecode 'l') according to the casting rule ''same_kind''
```

The above error is due to incompatible type, so it can be avoided by specifying the element type of the array, but the original array also has been updated,

```python
>>> a = np.array([[1, 2, 3], [4, 5, 6]], dtype = float)
>>> a1 = a[0, ]
>>> a1 /= 0.5
>>> a1
array([2., 4., 6.])
>>> a
array([[2., 4., 6.],
       [4., 5., 6.]])
```

In contrast, `/` would retain the original array.

```python
>>> a = np.array([[1, 2, 3], [4, 5, 6]], dtype = float)
>>> a1 = a[0, ]
>>> a1 = a1 / 0.5
>>> a1
array([2., 4., 6.])
>>> a
array([[1., 2., 3.],
       [4., 5., 6.]])
```

## `re`

### extract the starting position

Given an error message returns by `decode("utf8")` on a problematic characters, such as

```python
>>> "编程".encode("utf8")
b'\xe7\xbc\x96\xe7\xa8\x8b'
>>> b"\xe7\xbc".decode("utf8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
>>> e = "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"
```

I want to get the starting position. Suppose only one byte is wrong, and since a Chinese character is encoded by 3 bytes, then there are only two possible cases,

- `xoo` and `oox`: position `n`-`n+1`
- `oxo`: position `n`

```bash
>>> re.search("position ([0-9]{1,3})", e).group(1)
'0'
>>> re.search("position ([0-9]{1,3})", e).group(0)
'position 0'
```

where `()` is used to group, and `group(1)` returns the first group, while `group(0)` returns the whole match.

## 又拍云的短信平台

参考文献

1. [Python 使用requests发送POST请求 - CSDN博客](http://blog.csdn.net/junli_chen/article/details/53670887)
2. [Python-爬虫-requests库用语post登录](https://www.cnblogs.com/fredkeke/p/7000687.html)

## mdx_math安装命令

参考[manage-your-cms-using-mkdocs](http://wutongtree.github.io/devops/manage-your-cms-using-mkdocs)

```bash
sudo pip install python-markdown-math
```

## How can I use Conda to install MySQLdb?


参考[How can I use Conda to install MySQLdb?](https://stackoverflow.com/questions/34140472/how-can-i-use-conda-to-install-mysqldb)

## 远程连接 mysql

首先需要在服务器端，在`my.cnf` 中注释掉

```bash
# bind-address = 127.0.0.1
```

并且在 mysql 中创建用户并设置权限，如

```bash
create user 'test'@'%' identified by 'test123';
grant all privileges on testdb.* to 'test'@'%' with grant option;
```

参考

1. [Host 'xxx.xx.xxx.xxx' is not allowed to connect to this MySQL server](https://stackoverflow.com/questions/1559955/host-xxx-xx-xxx-xxx-is-not-allowed-to-connect-to-this-mysql-server)
2. [How to allow remote connection to mysql](https://stackoverflow.com/questions/14779104/how-to-allow-remote-connection-to-mysql)

## sphinx 相关

### __init__

[https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method](https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method)


## thread vs process

参考 [一道面试题：说说进程和线程的区别](https://foofish.net/thread-and-process.html)

```python
# -*- coding: utf-8 -*-

# https://foofish.net/thread-and-process.html

import os

# 进程是资源（CPU、内存等）分配的基本单位，它是程序执行时的一个实例。
# 程序运行时系统就会创建一个进程，并为它分配资源，然后把该进程放入进程就绪队列，
# 进程调度器选中它的时候就会为它分配CPU时间，程序开始真正运行。

print("current process: %s start..." % os.getpid())
pid = os.fork()
if pid == 0:
    print('child process: %s, parent process: %s' % (os.getpid(), os.getppid()))
else:
    print('process %s create child process: %s' % (os.getpid(), pid) )
    
# fork函数会返回两次结果，因为操作系统会把当前进程的数据复制一遍，
# 然后程序就分两个进程继续运行后面的代码，fork分别在父进程和子进程中返回，
# 在子进程返回的值pid永远是0，在父进程返回的是子进程的进程id。
    

# 线程是程序执行时的最小单位，它是进程的一个执行流，
# 是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，
# 线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。
```


## Kite 使用体验

Copilot 一直 detect 不出 spyder，只有刚开始装的时候检测到了，但那时候也没有起作用。而 kite 本身一直在 spyder 右下角的状态栏中。


## `pip`

- `pip list`: 查看已安装的包

### set mirror in mainland China

在win10下设置，参考[Python pip 国内镜像大全及使用办法](http://blog.csdn.net/testcs_dn/article/details/54374849)

在用户文件夹下新建pip文件夹，里面新建pip.ini文件

```txt
[global]
index-url=http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
```

注意编码格式为utf8无BOM。

### temporary proxy

通过 `conda` 安装镜像在 `.condarc` 中设置, 如在内地可以用[清华的镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)，而通过 `pip` 详见 [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)，临时使用可以运行

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

### upgrade package

```bash
# pip install PACKAGE -upgrade
pip install PACKAGE -U
```

### install from github

For example, ESL-CN used the forked plugin, [szcf-weiya/mkdocs-git-revision-date-plugin](https://github.com/szcf-weiya/mkdocs-git-revision-date-plugin), to set the update time, then in the `.travis.yml` file,

```bash
pip install git+https://github.com/szcf-weiya/mkdocs-git-revision-date-plugin.git
```

### 'Uninstalling a distutils installed project' error

```bash
pip install --ignore-installed ${PACKAGE_NAME}
```

refer to ['Uninstalling a distutils installed project' error when installing blockstack #504](https://github.com/blockstack/blockstack-core/issues/504)

## `sys`

### the first argument in `sys.path.insert()`

> But for `sys.path` specifically, element 0 is the path containing the script, and so using index 1 causes Python to search that path first and then the inserted path, versus the other way around when inserted at index 0.

The reason is that `sys.path` returns a list, while `.insert` is the method for a list, which insert object before the given index. Thus, if the first argument is 0, then the inserted path would be firstly searched, otherwise, it would be inserted after the first path, the current folder.

refer to [First argument of sys.path.insert in python](https://stackoverflow.com/questions/37176836/first-argument-of-sys-path-insert-in-python)

## TensorFlow

--8<-- "docs/TFnotes/README.md"

## the `i`-th row in pandas

```python
df_test.iloc[0]
```

## `key` in `sorted`

As said in [Key Functions -- Sorting HOW TO](https://docs.python.org/3/howto/sorting.html), the `key` function is to specify a function to be called on each list element prior to making comparisons.

```python
sorted("This is a test string from Andrew".split(), key=str.lower)
```

and met such technique in [4ment/marginal-experiments](https://github.com/4ment/marginal-experiments/blob/41124a1fbeed566cd7abc3dc474ea908a5ee8b28/run_simulations.py#L229)

## unittest and coverage

- candidate packages
	- [Coverage.py](https://coverage.readthedocs.io/en/coverage-5.5/)
	- [:white_check_mark: coveralls-python](https://coveralls-python.readthedocs.io/en/latest/usage/index.html)

- the schematic of unittest framework in python: [Running unittest with typical test directory structure](https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure)

### run locally
  
1. write unittest scripts in folder `test`
2. install [coveralls-python](https://coveralls-python.readthedocs.io/en/latest/usage/index.html): `conda install coveralls`
3. obtain the COVERALLS_REPO_TOKEN from [coveralls](https://coveralls.io/)
4. run the script file

=== "run_local_coverall.sh"
	```bash
	#!/bin/bash
	cd test/
	coverage run test_script.py
	mv .coverage ..
	cd ..

	COVERALLS_REPO_TOKEN=XXXXXXXXXXXXXXXXXXXXXXX coveralls
	```

### combine with julia in Actions

combine the coverage from julia by merging the resulted json file, which need `coveralls-lcov` to convert `LCOV` to `JSON`.

refer to:

- [An Example of Python in Github Action](https://github.com/aodj/icelandreview/actions/runs/33951158/workflow)
- [Multiple Language Support](https://coveralls-python.readthedocs.io/en/latest/usage/multilang.html)
- [GitHub: Cell-Video](https://github.com/szcf-weiya/Cell-Video/blob/102153462e7fd65718738bd2ad0ef37d4150f722/.github/workflows/blank.yml)


## Misc

- [人工鱼群算法-python实现](http://www.cnblogs.com/biaoyu/p/4857911.html)
- [请问phantom-proxy如何设置代理ip](https://segmentfault.com/q/1010000000685938)
- [Python编码介绍——encode和decode](http://blog.chinaunix.net/uid-27838438-id-4227131.html)
- [爬虫必备——requests](https://zhuanlan.zhihu.com/p/20410446)
- [Python使用代理抓取网站图片（多线程）](http://www.jb51.net/article/48112.htm)
- [python中threading模块详解（一）](http://blog.chinaunix.net/uid-27571599-id-3484048.html)
- [python 爬虫获取XiciDaili代理IP](http://30daydo.com/article/94)
- [使用SQLite](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001388320596292f925f46d56ef4c80a1c9d8e47e2d5711000)
- [python 使用sqlite3](http://www.cnblogs.com/hongten/p/hongten_python_sqlite3.html)
- [用Python进行SQLite数据库操作](http://www.cnblogs.com/yuxc/archive/2011/08/18/2143606.html)
- [Python调用MongoDB使用心得](https://www.oschina.net/question/54100_27233)
- [python urllib2详解及实例](http://www.pythontab.com/html/2014/pythonhexinbiancheng_1128/928.html)
- [Python爬虫入门实战七：使用Selenium--以QQ空间为例](https://www.jianshu.com/p/ffd02cc9d4ef)
- [Python中将打印输出导向日志文件](https://www.cnblogs.com/arkenstone/p/5727883.html)
- [python 中文编码(一)](https://www.cnblogs.com/tk091/p/4012004.html)
- [Python爬虫利器二之Beautiful Soup的用法](https://cuiqingcai.com/1319.html)
- [正则表达式之捕获组/非捕获组介绍](http://www.jb51.net/article/28035.htm)
- [Selenium using Python - Geckodriver executable needs to be in PATH](https://stackoverflow.com/questions/40208051/selenium-using-python-geckodriver-executable-needs-to-be-in-path)
