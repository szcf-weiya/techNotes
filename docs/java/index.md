## JDK, JRE and JVM

Refer to [JDK、JRE、JVM三者间的关系  ](http://playkid.blog.163.com/blog/static/56287260201372113842153/)

1. JDK(Java Development Kit): 针对开发员，包括了Java运行环境JRE、Java工具和Java基础类库
2. JRE(Java Runtime Environment): 运行JAVA程序所必须的环境的集合，包含JVM标准实现及Java核心类库
3. JVM(Java Virtual Machine): 能够运行以Java语言写作的软件程序。JVM屏蔽了与具体操作系统平台相关的信息，使得Java程序只需生成在Java虚拟机上运行的目标代码（字节码），就可以在多种平台上不加修改地运行。

## `array[index++]` vs `array[++index]`

> The code result++; and ++result; will both end in result being incremented by one. The only difference is that the prefix version (++result) evaluates to the incremented value, whereas the postfix version (result++) evaluates to the original value.

参考 [array index and increment at the same line](https://stackoverflow.com/questions/7218249/array-index-and-increment-at-the-same-line)

## Code Coverage of Java/Julia/R/Python

此前在 Python/Julia/R 中玩过简单的 code coverage，我的印象中基本分为两步

1. 生成测试文件
2. 上传至 coveralls/codecov 等 CI 网站

不同的语言有不同的 package 可以生成测试文件，并且上传至绑定好的第三方 CI 服务中。R 中

```bash
library(covr); codecov()
```

便可一键传至 codecov，比如 [szcf-weiya/fRLR](https://github.com/szcf-weiya/fRLR/blob/master/.travis.yml).

Julia 可以用 `Coverage` 包，然后通过 `submit` 函数提交至 CI 服务中，如 [szcf-weiya/GradientBoost.jl](https://github.com/szcf-weiya/GradientBoost.jl/blob/master/.travis.yml)

当然也可以先导出测试文件，`lcov.info`，然后再上传，目前我在 github action 是采用这种配置模式，而且通过在环境变量中加入 token，可以适用于 private 仓库。

Python 也类似有 `coverage` 包。

如果仓库中有混合代码，则可以先分别生成测试文件，再把 coverage 结果拼接起来一起上传至 CI 服务中，但是因为不同 package 输出的测试文件格式不一样，可能需要进行格式转换。比如在我的某个 Python + Julia 混合项目中，主要参考 [Multiple Language Support](https://coveralls-python.readthedocs.io/en/latest/usage/multilang.html)

- Julia 部分通过 `Coverage` 包导出 `lcov.info` 文件
- 通过 `coveralls-lcov` 将 `lcov.info` 转换成 JSON 格式 `lcov.json`
- python 部分通过 `coverage` 包进行测试，然后在用 `coveralls` 提交时，加上 `--merge=lcov.json` 的选项 

在 Java 中，也以为会有这种直接的方式，当然 Java 如果是采用 Maven 等框架的话，应该会很直接（因为看上去也是一句话的事），但是我的 Java 水平仅限于 HelloWorld，暂时不想也没必要折腾 Maven 等框架，而且我觉得从零开始把背后的逻辑弄清楚更有意义。

首先找到了单元测试 JUnit，这个符合我的预期，但是始终没能找到输出测试文件的选项，即只能在命令行跑完展示下。后来 google 试图找到直接从 JUnit 中导出测试结果的工具，但竟然没有，直到翻到[这个答案](https://stackoverflow.com/questions/12445582/generating-junit-reports-from-the-command-line)打消了我的念头，里面提到了 `Ant`，其实此时根本对 Ant 和 Maven 等等名词没有任何概念，即便看了官方的解释，也还是一头雾水。不管怎样，还是准备学下 `Ant`，一个[非常形象的类别——Java中的make](https://www.cnblogs.com/ArtsCrafts/p/Ant_begin.html)——瞬间让我理解了 `Ant` 的作用，然后接下来就像写 Makefile 一样来学着写 `build.xml`，所以在看官方文档之前，看看别人的总结也还是很有用的. 在 ant 中 task `junit` 可以指定输出选项，原以为这时大功告成，但是最后上传至 codecov 时，仍然表示解析报告出错，后来细看了了 `report.xml` 才发现里面根本包含不了 coverage 的信息，换言之，很可能 JUnit 只是起了 test 的作用，并不能分析 coverage。

这时便需要另外换工具了，其实在看到 JUnit 时，也看到了 `JaCoCo`，它也可以通过 ant 实现，只不过多了一些定义，所以不是很难。于是继续对照示例文档写 `build.xml`，得到的 `report.xml` 确实也包含了 coverage 的信息，但是上传至 codecov，仍然表示不能解析报告。最后跟示例文件比照 `build.xml` 以及 `report.xml`，发现我在 `report.xml` 中没有 sourcefiles 以及 classfiles 的信息，而这些是通过在 `build.xml` 中设置 `debug=true` 产生的。当把这个选项加上去，终于能在 codecov 看到 coverage 信息了！

顺带提一下混合代码的问题，对于 Java，似乎很难有方式让它与其他代码的测试结果拼起来，但是完全可以用两个不同的 CI 服务来展示不同语言代码的 coverage，比如 Java 放在 codecov，Julia 的放在 Coveralls.
