# PHP 相关

## 简介

> PHP 是一种创建动态交互性站点的强有力的服务器端脚本语言。

类比ASP.

使用[菜鸟工具](https://c.runoob.com/compile/1)进行在线测试。

另外有在线帮助手册：[中文](http://www.php.net/manual/zh/), [英文](http://www.php.net/manual/en/)

## 案例分析

下面的代码来自[fooleap](mailto: fooleap@gmail.com)的[disqus-php-api](https://github.com/fooleap/disqus-php-api)项目。

```php
<?php
/**
 * 批量获取评论数
 *
 * @param links  页面链接，以“,”分隔
 *
 * @author   fooleap <fooleap@gmail.com>
 * @version  2017-11-14 09:21:38
 * @link     https://github.com/fooleap/disqus-php-api
 *
 */
namespace Emojione;
require_once('init.php');
$links = '&thread=link:'.$website.preg_replace('/,/i','&thread=link:'.$website, $_GET['links']);
$fields_data = array(
    'api_key' => DISQUS_PUBKEY,
    'forum' => DISQUS_SHORTNAME,
    'limit' => 100
);
$curl_url = '/api/3.0/threads/list.json?'.http_build_query($fields_data).encodeURI($links);
$data = curl_get($curl_url);
$countArr = array();
foreach ( $data -> response as $key => $post ) {
    $countArr[$key] = array(
        'link'=> $post -> link,
        'posts'=> $post -> posts
    );
}
$output = $data -> code == 0 ? array(
    'code' => 0,
    'response' => $countArr
) : $data;
print_r(json_encode($output)); 
```

### 变量规则

1. 变量以 $ 符号开始，后面跟着变量的名称
2. 变量名必须以字母或者下划线字符开始
3. 变量名只能包含字母数字字符以及下划线（A-z、0-9 和 _ ）
4. 变量名不能包含空格
5. 变量名是区分大小写的（$y 和 $Y 是两个不同的变量）

### 数组 & 关联数组

参考[PHP 数组](http://www.runoob.com/php/php-arrays.html)

上述代码中

```php
$fields_data = array(
    'api_key' => DISQUS_PUBKEY,
    'forum' => DISQUS_SHORTNAME,
    'limit' => 100
);
```
即为关联数组。

而
```php
$countArr = array();
```
即为普通数组的创建。

### echo & print & print_r & var\_dump

参考[PHP 5 echo 和 print 语句](http://www.runoob.com/php/php-echo-print.html)

1. echo - 可以输出一个或多个字符串
2. print - 只允许输出一个字符串，返回值总为 1
3. echo 输出的速度比 print 快， echo 没有返回值，print有返回值1。
4. print\_r() - 可以把字符串和数字简单地打印出来，而数组则以括起来的键和值得列表形式显示，并以Array开头。但print\_r()输出布尔值和NULL的结果没有意义，因为都是打印"\n"。因此用var_dump()函数更适合调试。
5. var_dump(): 判断一个变量的类型与长度,并输出变量的数值,如果变量有值输的是变量的值并回返数据类型。此函数显示关于一个或多个表达式的结构信息，包括表达式的类型与值。数组将递归展开值，通过缩进显示其结构。

## 字符串连接

参考[关于php几种字符串连接的效率比较(详解)](http://www.jb51.net/article/106407.htm)，文章得出结论方法一效率最低。

1. 直接用.来进行连接。

2. 用.=进行连接。

3. 先压入数组，再通过join函数连接。

## cURL

需要安装`php-curl`，这依据php的版本而有所不同，具体参考[How to install php-curl in Ubuntu 16.04](https://stackoverflow.com/questions/38800606/how-to-install-php-curl-in-ubuntu-16-04/38801295)我采用的是php7，所以安装命令如下。

```bash
sudo apt-get install php-curl
```

**注意要重启apache。**

```php
<?php
// 创建一个新cURL资源
$ch = curl_init();

// 设置URL和相应的选项
curl_setopt($ch, CURLOPT_URL, "http://www.example.com/");
curl_setopt($ch, CURLOPT_HEADER, 0);

// 抓取URL并把它传递给浏览器
curl_exec($ch);

// 关闭cURL资源，并且释放系统资源
curl_close($ch);
?>
```


### curl_errno

```php
<?php
$error_codes=array(
[1] => 'CURLE_UNSUPPORTED_PROTOCOL', 
[2] => 'CURLE_FAILED_INIT', 
[3] => 'CURLE_URL_MALFORMAT', 
[4] => 'CURLE_URL_MALFORMAT_USER', 
[5] => 'CURLE_COULDNT_RESOLVE_PROXY', 
[6] => 'CURLE_COULDNT_RESOLVE_HOST', 
[7] => 'CURLE_COULDNT_CONNECT', 
[8] => 'CURLE_FTP_WEIRD_SERVER_REPLY',
[9] => 'CURLE_REMOTE_ACCESS_DENIED',
[11] => 'CURLE_FTP_WEIRD_PASS_REPLY',
[13] => 'CURLE_FTP_WEIRD_PASV_REPLY',
[14]=>'CURLE_FTP_WEIRD_227_FORMAT',
[15] => 'CURLE_FTP_CANT_GET_HOST',
[17] => 'CURLE_FTP_COULDNT_SET_TYPE',
[18] => 'CURLE_PARTIAL_FILE',
[19] => 'CURLE_FTP_COULDNT_RETR_FILE',
[21] => 'CURLE_QUOTE_ERROR',
[22] => 'CURLE_HTTP_RETURNED_ERROR',
[23] => 'CURLE_WRITE_ERROR',
[25] => 'CURLE_UPLOAD_FAILED',
[26] => 'CURLE_READ_ERROR',
[27] => 'CURLE_OUT_OF_MEMORY',
[28] => 'CURLE_OPERATION_TIMEDOUT',
[30] => 'CURLE_FTP_PORT_FAILED',
[31] => 'CURLE_FTP_COULDNT_USE_REST',
[33] => 'CURLE_RANGE_ERROR',
[34] => 'CURLE_HTTP_POST_ERROR',
[35] => 'CURLE_SSL_CONNECT_ERROR',
[36] => 'CURLE_BAD_DOWNLOAD_RESUME',
[37] => 'CURLE_FILE_COULDNT_READ_FILE',
[38] => 'CURLE_LDAP_CANNOT_BIND',
[39] => 'CURLE_LDAP_SEARCH_FAILED',
[41] => 'CURLE_FUNCTION_NOT_FOUND',
[42] => 'CURLE_ABORTED_BY_CALLBACK',
[43] => 'CURLE_BAD_FUNCTION_ARGUMENT',
[45] => 'CURLE_INTERFACE_FAILED',
[47] => 'CURLE_TOO_MANY_REDIRECTS',
[48] => 'CURLE_UNKNOWN_TELNET_OPTION',
[49] => 'CURLE_TELNET_OPTION_SYNTAX',
[51] => 'CURLE_PEER_FAILED_VERIFICATION',
[52] => 'CURLE_GOT_NOTHING',
[53] => 'CURLE_SSL_ENGINE_NOTFOUND',
[54] => 'CURLE_SSL_ENGINE_SETFAILED',
[55] => 'CURLE_SEND_ERROR',
[56] => 'CURLE_RECV_ERROR',
[58] => 'CURLE_SSL_CERTPROBLEM',
[59] => 'CURLE_SSL_CIPHER',
[60] => 'CURLE_SSL_CACERT',
[61] => 'CURLE_BAD_CONTENT_ENCODING',
[62] => 'CURLE_LDAP_INVALID_URL',
[63] => 'CURLE_FILESIZE_EXCEEDED',
[64] => 'CURLE_USE_SSL_FAILED',
[65] => 'CURLE_SEND_FAIL_REWIND',
[66] => 'CURLE_SSL_ENGINE_INITFAILED',
[67] => 'CURLE_LOGIN_DENIED',
[68] => 'CURLE_TFTP_NOTFOUND',
[69] => 'CURLE_TFTP_PERM',
[70] => 'CURLE_REMOTE_DISK_FULL',
[71] => 'CURLE_TFTP_ILLEGAL',
[72] => 'CURLE_TFTP_UNKNOWNID',
[73] => 'CURLE_REMOTE_FILE_EXISTS',
[74] => 'CURLE_TFTP_NOSUCHUSER',
[75] => 'CURLE_CONV_FAILED',
[76] => 'CURLE_CONV_REQD',
[77] => 'CURLE_SSL_CACERT_BADFILE',
[78] => 'CURLE_REMOTE_FILE_NOT_FOUND',
[79] => 'CURLE_SSH',
[80] => 'CURLE_SSL_SHUTDOWN_FAILED',
[81] => 'CURLE_AGAIN',
[82] => 'CURLE_SSL_CRL_BADFILE',
[83] => 'CURLE_SSL_ISSUER_ERROR',
[84] => 'CURLE_FTP_PRET_FAILED',
[84] => 'CURLE_FTP_PRET_FAILED',
[85] => 'CURLE_RTSP_CSEQ_ERROR',
[86] => 'CURLE_RTSP_SESSION_ERROR',
[87] => 'CURLE_FTP_BAD_FILE_LIST',
[88] => 'CURLE_CHUNK_FAILED');

?>
```