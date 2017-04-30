# 简介
GCC: GNU C Compiler
support: 
	1. C
	2. Ada
	3. C++
	4. Java
	5. Objective C
	6. Pascal
	7. COBOL
	
# 简单编译
test.c

一步到位的编译指令为

```gcc
gcc test.c -o test
```
## 预处理
```gcc
gcc -E test.c -o test.i
// gcc -E test.c
```
-E 使得编译器在预处理后停止，并输出预处理结果
## 编译为汇编代码
```gcc
gcc -S test.i -o test.s
```
-S 程序编译期间，生成汇编代码后，停止
## 汇编
```gcc
gcc -c test.s -o test.o
```
## 连接
```gcc
gcc test.o -o test
```

# 多个程序文件的编译
```gcc
gcc test1.c test2.c -o test 
```

# 检错
```gcc
gcc -pedantic illcode.c -o illcode
gcc -Wall illcode.c -o illcode
gcc -Werror test.c -o test
```

# 库文件连接
函数库包含.h,so,lib,dll

## 编译成可执行文件
```gcc
gcc –c –I /usr/dev/mysql/include test.c –o test.o
```
## 连接
```gcc
gcc –L /usr/dev/mysql/lib –lmysqlclient test.o –o test
```

## 强制连接时使用静态链接库

优先使用动态链接库，只有当动态链接库不存在时才考虑使用静态链接库，-static强制使用静态链接库

在/usr/dev/mysql/lib目录下有链接时所需要的库文件libmysqlclient.so和libmysqlclient.a，为了让GCC在链接时只用到静态链接库，可以使用下面的命令

```gcc
gcc –L /usr/dev/mysql/lib –static –lmysqlclient test.o –o test
```

### 静态库链接时搜索路径顺序:
1. ld会去找GCC命令中的参数-L
2. 再找gcc的环境变量LIBRARY_PATH
3. 再找内定目录 /lib /usr/lib /usr/local/lib 这是当初compile gcc时写在程序内的

### 动态链接时、执行时搜索路径顺序:
1. 编译目标代码时指定的动态库搜索路径
2. 环境变量LD\_LIBRARY\_PATH指定的动态库搜索路径
3. 配置文件/etc/ld.so.conf中指定的动态库搜索路径
4. 默认的动态库搜索路径/lib
5. 默认的动态库搜索路径/usr/lib

### 环境变量
LIBRARY\_PATH环境变量：指定程序静态链接库文件搜索路径
LD\_LIBRARY_PATH环境变量：指定程序动态链接库文件搜索路径
