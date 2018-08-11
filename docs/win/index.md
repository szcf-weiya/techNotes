# win 问题及解决

## win10 应用商店、照片等程序打不开

参考[win10 应用商店误删 求修复方法](https://answers.microsoft.com/zh-hans/windows/forum/windows_10-windows_store/win10/666838b7-7acd-4455-9217-bb0d92577941?auth=1)


管理员在命令行中运行
```
Get-AppXPackage -AllUsers | Foreach {Add-AppxPackage -DisableDevelopmentMode -Register "$($_.InstallLocation)\AppXManifest.xml"}
```

## word 在试图打开文件时遇到错误

[word 在试图打开文件时遇到错误](https://answers.microsoft.com/zh-hans/msoffice/forum/msoffice_word-mso_other-mso_archive/word/44473bde-599b-4552-99b1-0282e9ffe66e?messageId=1a74ab7c-2705-4db3-9f81-b58817a7a731)

## HP打印机run dll找不到模块

参考[Question](https://h30434.www3.hp.com/t5/LaserJet-Printing/There-was-a-problem-starting-C-Program-Files-HP-HP/m-p/2663133/highlight/true#M93469)

进入
"c:\users\*username*\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\"

中的 Monitor Ink Alerts - HP Photosmart 5510 series.lnk or anything HP printer related. Delete it.

## vs2015配置opencv

[OpenCV 2.4.13 在 VS2015上的配置](http://blog.csdn.net/lfw198911/article/details/52649459)

## Windows下安装MongoDB 3.2

[Windows下安装MongoDB 3.2](http://blog.csdn.net/u012995964/article/details/50943916)

[MongoDB入门](http://www.cnblogs.com/huangxincheng/archive/2012/02/18/2356595.html)

## Windows 10 安装 rJava

1. windows 10
2. R 3.3.0
3. jdk 1.8.0

成功安装rJava需要jvm.dll这个文件，所以很简单的做法就是把这个文件的路径添加到环境变量中，比如该文件在我电脑上的安装目录为

> C:\Program Files\Java\jdk1.8.0_102\jre\bin\server

你把这个路径加到环境变量中的PATH中就ok了。

# 安装 okular

1. 先安装 KDE 包管理器 chocolatey，[The package manager for Windows](https://chocolatey.org/)
2. 再安装 okular，[choco install okular](https://chocolatey.org/search?q=okular)