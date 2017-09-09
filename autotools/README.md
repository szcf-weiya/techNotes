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
