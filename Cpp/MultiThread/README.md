# 多线程编程（未完）

参考http://blog.csdn.net/hitwengqi/article/details/8015646

# say_hello

两次运行的结果混乱，因为没有同步？

![](res_1.png)

![](res_2.png)


# say_hello_paras

![](res_3.png)

结果混乱！！

可能原因：主进程还没开始对i赋值，线程已经开始跑了...?

# say_hello_paras_revise

![](res_4.png)
![](res_5.png)
