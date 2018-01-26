# python 相关

## 使用ipython %matplotlib inline

参考[cnblog](http://blog.csdn.net/u010194274/article/details/50698514)

使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。inline表示将图表嵌入到Notebook中

## seaborn的使用

Matplotlib自动化程度非常高，但是，掌握如何设置系统以便获得一个吸引人的图是相当困难的事。为了控制matplotlib图表的外观，Seaborn模块自带许多定制的主题和高级的接口。
[segmentfault](https://segmentfault.com/a/1190000002789457)


## 远程访问jupyter
[远程访问jupyter](http://www.cnblogs.com/zhanglianbo/p/6109939.html)

## jupyter notebook 出错

![](error_jupyter.png)

可以通过
```
rm -r .pki
```
解决


## 创建jupyter notebook 权限问题

![](error_jupyter_1.png)

原因是所给的路径的用户权限不一致，jupyter的用户及用户组均为root，为解决这个问题，直接更改用户权限

```
sudo chown weiya jupyter/ -R
sudo chgrp weiya jupyter/ -R
```
其中-R表示递归调用，使得文件夹中所有内容的用户权限都进行更改。


## theano import出错

![](err_theano.png)

更改.theano文件夹的用户权限

## python 查看已安装的包

```
pip list
```

## conda更新spyder

```
(sudo proxychains) conda update spyder
```

## selenium

refer to [Selenium using Python - Geckodriver executable needs to be in PATH
](https://stackoverflow.com/questions/40208051/selenium-using-python-geckodriver-executable-needs-to-be-in-path)

## array operation

```
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

## Difference between numpy.array shape (R, 1) and (R,)

refer to [Difference between numpy.array shape (R, 1) and (R,)](https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r)

## 正式认识conda

参考[https://conda.io/docs/user-guide/getting-started.html](https://conda.io/docs/user-guide/getting-started.html)

## 为py3安装spyder

1. 先建一个conda环境bunny，安装python3.4，因为要支持pyside，而经试验3.5+不支持。
2. 安装cmake
3. pyside出现keyerrror

转向py3.6
不装pyside，而装pyqt5
```bash
pip install pyqt5
```

## xrange and range

参考[Python中range和xrange的区别](http://blog.csdn.net/imzoer/article/details/8742283)

## 安装sqlite3

参考[Python安装sqlite3](http://www.python88.com/topic/420/)

## windows下安装lxml

参考[python安装lxml，在windows环境下](http://blog.csdn.net/g1apassz/article/details/46574963)

## 缺少Python27_d.lib

参考[缺少Python27_d.lib的解决方法](http://blog.csdn.net/junparadox/article/details/52704287)

## 远程访问Jupyter Notebook

[远程访问Jupyter Notebook](http://www.cnblogs.com/zhanglianbo/p/6109939.html)

## 人工鱼群算法-python实现

[人工鱼群算法-python实现](http://www.cnblogs.com/biaoyu/p/4857911.html)


## 请问phantom-proxy如何设置代理ip

[请问phantom-proxy如何设置代理ip](https://segmentfault.com/q/1010000000685938)

## python 编码介绍

[Python编码介绍——encode和decode](http://blog.chinaunix.net/uid-27838438-id-4227131.html)

## 爬虫必备requests

[爬虫必备——requests](https://zhuanlan.zhihu.com/p/20410446)

## Python使用代理抓取网站图片（多线程）

[Python使用代理抓取网站图片（多线程）](http://www.jb51.net/article/48112.htm)

## python中threading模块详解

[python中threading模块详解（一）](http://blog.chinaunix.net/uid-27571599-id-3484048.html)

## python 爬虫获取XiciDaili代理IP

[python 爬虫获取XiciDaili代理IP](http://30daydo.com/article/94)

## 使用SQLite

[使用SQLite](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001388320596292f925f46d56ef4c80a1c9d8e47e2d5711000)

[python 使用sqlite3](http://www.cnblogs.com/hongten/p/hongten_python_sqlite3.html)

[用Python进行SQLite数据库操作](http://www.cnblogs.com/yuxc/archive/2011/08/18/2143606.html)

[Python调用MongoDB使用心得](https://www.oschina.net/question/54100_27233)

## python urllib2详解及实例

[python urllib2详解及实例](http://www.pythontab.com/html/2014/pythonhexinbiancheng_1128/928.html)

## 使用Selenium

参考[Python爬虫入门实战七：使用Selenium--以QQ空间为例](https://www.jianshu.com/p/ffd02cc9d4ef)

## Python中将打印输出导向日志文件

参考[Python中将打印输出导向日志文件](https://www.cnblogs.com/arkenstone/p/5727883.html)


## python 中文编码

参考[python 中文编码(一)](https://www.cnblogs.com/tk091/p/4012004.html)

## Python爬虫利器二之Beautiful Soup的用法

参考[Python爬虫利器二之Beautiful Soup的用法](https://cuiqingcai.com/1319.html)


