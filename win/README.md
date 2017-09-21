# win 问题及解决

## win10 应用商店、照片等程序打不开

参考[win10 应用商店误删 求修复方法 ](
https://answers.microsoft.com/zh-hans/windows/forum/windows_10-windows_store/win10/666838b7-7acd-4455-9217-bb0d92577941?auth=1)


管理员在命令行中运行
```
Get-AppXPackage -AllUsers | Foreach {Add-AppxPackage -DisableDevelopmentMode -Register "$($_.InstallLocation)\AppXManifest.xml"}
```

## word 在试图打开文件时遇到错误

[word 在试图打开文件时遇到错误](https://answers.microsoft.com/zh-hans/msoffice/forum/msoffice_word-mso_other-mso_archive/word/44473bde-599b-4552-99b1-0282e9ffe66e?messageId=1a74ab7c-2705-4db3-9f81-b58817a7a731)