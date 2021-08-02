# Autotools

![](diagram.png)

关于Autotools有个很棒的[基础知识介绍文档](https://devmanual.gentoo.org/general-concepts/autotools/index.html)

## 第一篇.ac
```
dnl package name and version
AC_INIT([depi], 0.2.0)

dnl check gsl lib
: ${GSL_LIBS=`gsl-config --libs`}
if test -n "${GSL_LIBS}"; then
  LIBS="${GSL_LIBS}"
else
  echo "could find gsl"
  exit 1
fi

AC_SUBST(LIBS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
```

几个注意点
1. 等号附近不要有空格
2. 想要实现在命令行中运行`gsl-config --libs`的效果，请参照上述代码的实现。

## Make

[https://my.oschina.net/u/1413984/blog/199029](https://my.oschina.net/u/1413984/blog/199029)

- `$@`: 目标文件，比如 [gsl_lm/ols/Makefile](https://github.com/szcf-weiya/gsl_lm/blob/86d8c4846ed56a27ad8a9f35d9f1229fab704912/ols/Makefile#L22)
- `$^`: 所有的依赖文件，比如 [G-squared/src/Makefile](https://github.com/szcf-weiya/G-squared/blob/4f70c3f735e4241f7ba33986c9b6a53fdd0dc6ea/src/Makefile#L9-L21)
- `$<`: 第一个依赖文件

and [Makefile 经典教程(掌握这些足够)](http://blog.csdn.net/ruglcc/article/details/7814546/)

- define functions: e.g., [`mk` for compiling different tex files](https://github.com/szcf-weiya/Clouds/blob/fbbc42953e724818e3ce0c727efbe457e5081e68/notes/Makefile#L1-L7)