# JDK, JRE and JVM

Refer to [JDK、JRE、JVM三者间的关系  ](http://playkid.blog.163.com/blog/static/56287260201372113842153/)

1. JDK(Java Development Kit): 针对开发员，包括了Java运行环境JRE、Java工具和Java基础类库
2. JRE(Java Runtime Environment): 运行JAVA程序所必须的环境的集合，包含JVM标准实现及Java核心类库
3. JVM(Java Virtual Machine): 能够运行以Java语言写作的软件程序。JVM屏蔽了与具体操作系统平台相关的信息，使得Java程序只需生成在Java虚拟机上运行的目标代码（字节码），就可以在多种平台上不加修改地运行。

# `array[index++]` vs `array[++index]`

> The code result++; and ++result; will both end in result being incremented by one. The only difference is that the prefix version (++result) evaluates to the incremented value, whereas the postfix version (result++) evaluates to the original value.

参考 [array index and increment at the same line](https://stackoverflow.com/questions/7218249/array-index-and-increment-at-the-same-line)