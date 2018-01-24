# Cpp notes

## C++中cout输出字符型指针地址值的方法
[ref](http://www.cnblogs.com/wxxweb/archive/2011/05/20/2052256.html)

## const 总结
[ref](http://www.2cto.com/kf/201210/160536.html)

## new

申请空间，并执行相应的构造函数

## delete

执行析构函数，并释放空间

引用的本质是指针常量

```
const int m;
//int* p = &m;//wrong
const int* p = &m;
int *const pc = &m; //必须初始化，引用的本质
```


构造函数析构函数作用区间

```
A a;
A *ap;
if (...)
{
	B b;
	...// B析构
	ap = new A;
}
......// A析构
delete ap;
```


先执行基类构造函数，再派生类构造函数；
先执行派生类析构函数，再派生基类析构函数。


## 函数模板和模板函数

[ref](http://blog.csdn.net/beyondhaven/article/details/4204345)

C++中，函数模板与同名的非模板函数重载时，应遵循下列调用原则：
1. 寻找一个参数完全匹配的函数，若找到就调用它。若参数完全匹配的函数多于一个，则这个调用是一个错误的调用。
2. 寻找一个函数模板，若找到就将其实例化生成一个匹配的模板函数并调用它。
3. 若上面两条都失败，则使用函数重载的方法，通过类型转换产生参数匹配，若找到就调用它。
4. 若上面三条都失败，还没有找都匹配的函数，则这个调用是一个错误的调用。

## 初始化列表

[ref](http://www.cnblogs.com/graphics/archive/2010/07/04/1770900.html)


## extern

[reference](http://www.cnblogs.com/yc_sunniwell/archive/2010/07/14/1777431.html)

在Rcpp中，extern "C" 告诉编译器，保持其名称，不要生成用于链接的中间函数名。

## "symbol lookup error"

```
./test: symbol lookup error: ./test: undefined symbol:
```

动态链接库的原因，因为更新完gsl之后，原先的动态链接库不管用了，可以用下面的命令追踪动态链接库
```
ldd test
ldd -d -r test
```

参考[c++ runtime "symbol lookup error" ](http://gdwarner.blogspot.com/2009/03/c-runtime-symbol-lookup-error.html)

## 字符数组与数字互换

http://blog.csdn.net/sunquana/article/details/14645079

### 字符数字转数字
1. atoi
2. atof
3. atol
4. strtod
5. strtol

### 数字转字符
sprintf

## 指针初始化

```
double x;
double *p = &x;
```

DO NOT
```
double *p = 5;
```

BUT
```
double *p = "aaa";
```
并且要初始化，不能

```
double *p;
```
然后直接传参了，这是不对的。

## C: Correctly freeing memory of a multi-dimensional array

https://stackoverflow.com/questions/1733881/c-correctly-freeing-memory-of-a-multi-dimensional-array

## \#pragma unroll的用法
http://blog.csdn.net/fengzizhuang/article/details/9300431

## Getting std :: ifstream to handle LF, CR, and CRLF?

1. https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf

2. https://stackoverflow.com/questions/9188126/c-strange-behavior-with-stdistream-or-sentry-wrap-around/9189541#9189541

```
std::istream& safeGetline(std::istream& is, std::string& t)
{
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if(sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if(t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}
```

使用这个代码注意一个问题
应该使用
```
while(!safeGetline(input, line).eof)
```

不能用
```
while(safeGetline(input, line))
```

## fPIC

参考http://blog.sina.com.cn/s/blog_54f82cc201011op1.html

## 参数和返回值的三种传递方式

[C++函数参数和返回值三种传递方式：值传递、指针传递和引用传递（着重理解）](http://blog.csdn.net/thisispan/article/details/7456180)

## linux gcc常用命令

[Linux GCC常用命令](http://www.cnblogs.com/ggjucheng/archive/2011/12/14/2287738.html)
