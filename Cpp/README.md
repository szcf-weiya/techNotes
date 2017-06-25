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
