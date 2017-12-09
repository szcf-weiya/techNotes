# 微信小程序开发记录

注意后台服务器的域名需要备案并且需要购买证书实现https，短期不行。幸运的是，hotapp提供了替代方案。

## 参考网址

1. 官方文档：https://mp.weixin.qq.com/debug/wxadoc/dev/
2. HotApp：https://weixin.hotapp.cn/
3. http://www.jianshu.com/p/38d13a7c1b78
4. enable https: https://certbot.eff.org/#ubuntuxenial-nginx

## 配置nginx

1. http://www.jianshu.com/p/132eed84bc4f
2. https://www.cnblogs.com/erbingbing/p/7220645.html?utm_source=itdadao&utm_medium=referral

  很简单，只需要
```
sudo apt-get install nginx
```

接着在浏览器中输入网址，便可以看到Welcome界面。

## 修改Nginx web服务器根目录

https://yq.aliyun.com/ziliao/91831?spm=5176.8246799.0.0.DH1Qgn

## 安装nodejs

https://help.aliyun.com/document_detail/50775.html?spm=5176.doc25426.6.655.kn1mB7

测试：

可以在任意位置编写example.js，注意端口号不能仍用80.

不要对静态网页进行post，post的时候加上相对应的端口。

## nodejs调用python脚本

nodejs 收发数据

https://www.cnblogs.com/gamedaybyday/p/6637933.html

# 腾讯云折腾

1. python main.py 80 报错，原因应该是普通用户不行，需要加上sudo，而8080口普通用户可直接访问。参考 https://www.cnblogs.com/lwhp/p/5789305.html

# 443 端口转发到3000端口

https://stackoverflow.com/questions/42767106/how-to-run-nodejs-server-over-443-ensuring-nginx-doesnt-stop-working

同样的方法对于80端口似乎不管用，可能原因

1. 之前操作过一次iptables，参见https://eladnava.com/binding-nodejs-port-80-using-nginx/。
2. 其他原因

另外注意用https的时候，不要直接写IP，不然python会报错
```
SSLError: hostname '112.74.43.59' doesn't match 'seminar.hohoweiya.xyz'
```
