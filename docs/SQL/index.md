# SQL相关

## SQLite连接字符串

参考[SQLite 连接两个字符串](http://www.cnblogs.com/AngelLee2009/p/3208223.html)

SQLite中，连接字符串不是使用+，而是使用||

```sql
SELECT 'I''M '||'Chinese.' 
```

notes:
1. 
```sql
SELECT 'I''M '+'Chinese.' 
```
将输出0.
2. 默认情况下, '是字符串的边界符, 如果在字符串中包含', 则必须使用两个', 第1个'就是转义符。

## 取得sqlite数据库里所有的表名 &复制表

[取得sqlite数据库里所有的表名 &复制表](http://blog.csdn.net/vlily/article/details/9096909)


