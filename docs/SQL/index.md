# SQL 笔记

- [廖雪峰：SQL 教程](https://www.liaoxuefeng.com/wiki/1177760294764384)

此前一直有个困惑，本科选修了一门跟数据库有关的课程，印象中关键字都是大写的，然而碰到的 MySQL 语句都是小写的。原来，

> SQL语言关键字不区分大小写！！！但是，针对不同的数据库，对于表名和列名，有的数据库区分大小写，有的数据库不区分大小写。同一个数据库，有的在Linux上区分大小写，有的在Windows上不区分大小写。

## 关系模型

### 主键

原则：不使用任何业务相关的字段作为主键。如身份证号、手机号、邮箱地址。

作为主键最好是完全业务无关的字段，如

- 自增整数类型
- 全局唯一 [GUID/UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier) 类型。

联合主键：两个及其以上的字段都设为主键。

### 外键

外键通过定义约束实现，

```sql
ALTER TABLE students
ADD CONSTRAINT fk_class_id
FOREIGN KEY (class_id)
REFERENCES classes (id);
```

其中约束的名称 `fk_class_id` 可以任意，第三行指定了 `class_id` 作为外键，而第四行指定了这个外键将关联到 `classes` 表的主键 `id`.

- 外键约束可以保证无法插入无效的数据。如果 `classes` 表中不存在 `id=99` 的记录，则 `students` 表也就无法插入 `class_id=99` 的记录。
- 外键约束会降低数据库的性能，大部分互联网应用程序为了追求速度，并不设置外键约束，而是仅靠应用程序自身来保证逻辑的正确性。

删除外键约束：

```sql
ALTER TABLE students
DROP FOREIGN KEY fk_class_id;
```

这并不会删除外键这一列。

### 索引

索引是关系数据库中对某一列或多个列的值进行预排序的数据结构。通过使用索引，可以让数据库系统不必扫描整个表，而是直接定位到符合条件的记录，这样就大大加快了查询速度。原理参见 [知乎：深入浅出数据库索引原理](https://zhuanlan.zhihu.com/p/23624390)。

```sql
ALTER TABLE students
ADD INDEX idx_score (score);
```

这使用列 `score` 的名称为 `idx_score` 的索引。

- 索引的效率取决于索引列的值是否散列，即该列的值如果越互不相同，那么索引效率越高。
- 索引的优点是提高了查询效率，缺点是在插入、更新和删除记录时，需要同时修改索引，因此，索引越多，插入、更新和删除记录的速度就越慢。
- 关系数据库会自动对其创建主键索引。使用主键索引的效率是最高的，因为主键会保证绝对唯一。

另外可以指定 `UNIQUE` 来创建唯一索引，

```sql
ALTER TABLE students
ADD UNIQUE INDEX uni_name (name);
```

也可以只对某一列添加一个唯一的约束而不创建索引。

```sql
ALTER TABLE students
ADD CONSTRAINT uni_name UNIQUE (name);
```

## Operations on Databases

### create

the script is downloaded from [init-test-data.sql](https://raw.githubusercontent.com/michaelliao/learn-sql/master/mysql/init-test-data.sql)

```bash
$ mysql -u root -p < init-test-data.sql 
Enter password: 
result:
ok
```

!!! warning
    The creation of databases require `root`, and other users would fail.
    > ERROR 1044 (42000) at line 2: Access denied for user 'weiya'@'%' to database 'test'

### grant privileges

we can check the privileges of one user

```sql
mysql> show grants for 'weiya'@'%';
+------------------------------------------------------------------------------------------------------+
| Grants for weiya@%                                                                                   |
+------------------------------------------------------------------------------------------------------+
| GRANT USAGE ON *.* TO 'weiya'@'%' IDENTIFIED BY PASSWORD '******************' |
| GRANT ALL PRIVILEGES ON `FruitCupdb`.* TO 'weiya'@'%' WITH GRANT OPTION                              |
+------------------------------------------------------------------------------------------------------+
2 rows in set (0.00 sec)
```

where the `USAGE` in the first grant just means `no privileges`, see [Why is a “GRANT USAGE” created the first time I grant a user privileges?](https://stackoverflow.com/questions/2126225/why-is-a-grant-usage-created-the-first-time-i-grant-a-user-privileges) for more details.

- grant all privileges to a user on a database

```sql
mysql> grant all privileges on test.* to 'weiya'@'%';
```

where `test.*` means `database_name.table_name`.

### query

- 基本查询：返回所有行和列：`select * from test.students;`
    - 不带 `from` 子句的 `select` 语句可以用于计算，`select 100+100;`，或者用于判断当前到数据库的连接是否有效，比如用`select 1;`。
- 条件查询：`select * from <table> where [NOT] condition1 AND/OR condition2;`
- 投影查询：`select col1, col2 from ...;`
    - 可以采用空格对列名进行重命名：`select col1 newcol1, col2 from ...;`
- 排序: `select col1 from ... ordered by colx [DESC] [, coly];` 其中 `DESC` 表示降序
- 分页查询： `select ... limit <M> offset <N>;` 其中 `offset` 表示开始计数的 index，**SQL记录集的索引从0开始**，`limit` 表示最多的条数。
- 聚合查询：`select count(*) num from students;`, 其中 `num` 为设置的别名。
    - 其它聚合函数，`SUM()`, `AVG()`, `MAX()`, `MIN()`,前两者要求数值类型，而后两个可以用于字符串
    - 如果 `where` 条件没有匹配到任何行，则 `count()` 返回 0，而其它四个聚合函数返回 `NULL`。
- 多表查询，比如

```sql
SELECT
    s.id sid,
    s.name,
    s.gender,
    s.score,
    c.id cid,
    c.name cname
FROM students s, classes c
WHERE s.gender = 'M' AND c.id = 1;
```

- 连接 (join) 查询：先确定主表，然后把另一个表的数据附加到结果集上
    - inner join: 两张表都存在的记录
    - left/right outer join: 左（右）表存在的记录，不存在的填充为 NULL
    - full outer join: 左右表都存在的记录

### modify

- insert

```sql
INSERT INTO <表名> (字段1, 字段2, ...) VALUES (值1, 值2, ...);
```

- update

```sql
UPDATE <表名> SET 字段1=值1, 字段2=值2, ... WHERE ...;
```

- delete

```sql
DELETE FROM <表名> WHERE ...;
```

### manage

- `show databases;`
- `create/drop database XX;`
- `use XX;`
- `show tables;`
- `create/drop table XX;`
- 查看表的结构：`DESC XX;`
- 查看创建表的sql语句：`show create table XX;`
- 新增列：`alter table XX add column XX VARCHAR(10) NOT NULL;`
- 修改列：`alter table XX change column XX YY VARCHAR(20) NOT NULL;`
- 删除列：`alter table XX drop column XX;`

### other tips

- 插入或替换：`REPLACE INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99);`
- 插入或更新：`INSERT INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99) ON DUPLICATE KEY UPDATE name='小明', gender='F', score=99;`
- 插入或忽略：`INSERT IGNORE INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99);`
- 快照：`CREATE TABLE students_of_class1 SELECT * FROM students WHERE class_id=1;`
- 写入查询结果集：`INSERT INTO statistics (class_id, average) SELECT class_id, AVG(score) FROM students GROUP BY class_id;`
- 强制使用指定索引：`SELECT * FROM students FORCE INDEX (idx_class_id) WHERE class_id = 1 ORDER BY id DESC;`

### 事务

> 这种把多条语句作为一个整体进行操作的功能，被称为数据库事务。数据库事务可以确保该事务范围内的所有操作都可以全部成功或者全部失败。如果事务失败，那么效果就和没有执行这些SQL一样，不会对数据库数据有任何改动。

## MySQL

### Installation

```bash
$ sudo apt-get install mysql-server
```

大概是在做[FruitCup](https://github.com/szcf-weiya/FruitCup)时就已经在 Aliyun 的 Ubuntu 14.04 上装好了 MySQL，其版本为

```bash
$ mysql --version
mysql  Ver 14.14 Distrib 5.5.62, for debian-linux-gnu (x86_64) using readline 6.3
```

在 WSL (Ubuntu 18.04.1 LTS) 中通过 `TAB` 键，可见备选可以安装的与 MySQL 有关的包有

```bash
$ sudo apt-get install mysql
mysql-client           mysql-server-5.7       mysql-workbench
mysql-client-5.7       mysql-server-core-5.7  mysql-workbench-data
mysql-client-core-5.7  mysql-source-5.7       mysqltcl
mysql-common           mysql-testsuite        mysqltuner
mysql-sandbox          mysql-testsuite-5.7    
mysql-server           mysql-utilities  
```

在 T460P (Ubuntu 18.04) 上只装客户端，

```bash
~$ sudo apt install mysql-client-5.7
~$ mysql --version
mysql  Ver 14.14 Distrib 5.7.33, for Linux (x86_64) using  EditLine wrapper
```

### Login in

- login with explicit password: `mysql -u USERNAME -pXXXX`, where no spaces after `-p`
- enter password invisibly: `mysql -u USERNAME -p`
- login from local: `mysql -h IP -u USERNAME -p`, where the user requires to be accessed from `%` host, and make sure no firewall on the 3306 port.

注意到，`mysql` 实际上是 MySQL 客户端，而真正的 MySQL 服务器程序是 mysqld. 在 Client 端输入的 SQL 语句通过 TCP 连接发送到 MySQL Server。默认端口号是 3306，即如果发送到本机的 MySQL Server, 地址即为 `127.0.0.1:3306`.

### Forget root password

Follow the steps in [Step by step guide to reset root password in Mysql](https://linuxtechlab.com/reset-root-password-mysql/)

```bash
# login as root
$ su
$ service mysql stop # change to mysqld for centos
$ mysqld_safe –skip-grant-tables & # the option is NOT recommended & should only be done to reset the password
# login to mysql
$ mysql
# update password
mysql> UPDATE mysql.user SET Password=PASSWORD('updated-password') WHERE User='root';
# exit safely
mysql> flush privileges;
mysql> exit;
# restart mysql 
$ service mysql restart
```

### Manage users

- list all mysql user accounts

```sql
mysql> select user, host from mysql.user;
+------------------+-----------+
| user             | host      |
+------------------+-----------+
| weiya            | %         |
| root             | 127.0.0.1 |
```

where `%` means the user `weiya` can login the mysql from anywhere.

- change user password: similar as for the above root user

```sql
mysql> update mysql.user set password=PASSWORD('XXXXXXXXX') where User='weiya';
```

refer to 
- [How to Create MySQL Users Accounts and Grant Privileges](https://linuxize.com/post/how-to-create-mysql-user-accounts-and-grant-privileges)
- [How to Manage MySQL Databases and Users from the Command Line](https://linuxize.com/post/how-to-manage-mysql-databases-and-users-from-the-command-line/)

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