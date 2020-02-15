# ipv6 on aws

在 aws 中新建了一个 EC2，但是默认没有 ipv6 address，而看到[官方文档说应该在创建的时候额外配置一下](https://docs.amazonaws.cn/en_us/AWSEC2/latest/UserGuide/using-instance-addressing.html)，但是底下又说道新建完后依然可以添加 ipv6，然而在 “Actions, Networking, Manage IP Addresses” 并没有看到所说的 “IPv6 Addresses”。另外有尝试过按照文档重新创建一个 EC2，然而并不行。所以还是决定在已经创建好的那个 EC2 上折腾。

我的猜测是需要配置什么东西，才会在 “Actions, Networking, Manage IP Addresses” 中出现 “IPv6 Addresses”，所以摸索过程中我以这个为目标看看我有没有配对。

## Update VPC

首先在 [AWS IPv6 Update – Global Support Spanning 15 Regions & Multiple AWS Services](https://aws.amazon.com/blogs/aws/aws-ipv6-update-global-support-spanning-15-regions-multiple-aws-services/) 中确认了我的所在区域是支持 ipv6 的。然后看到 [New – IPv6 Support for EC2 Instances in Virtual Private Clouds](https://aws.amazon.com/blogs/aws/new-ipv6-support-for-ec2-instances-in-virtual-private-clouds/) 这里介绍在 VPC 中设置。于是通过 instance 页面点击其 VPC，进入 VPC 设置页面，在 CIDR BLOCKs 中添加 IPv6 CIDR Blocks, 像下图一样选择 `Amazon provided IPv6 CIDR block`

![](https://media.amazonwebservices.com/blog/2016/vpc_ipv6_create_vpc_1.png)

## Update subnet

仅仅设置了 VPC 的 CIDR blocks 还不够，需要设置对应的 subnet 的 IPv6 CIDR，注意这里会让我们填写ipv6 address末尾几位，我一开始什么都没填直接点确认，会报错，然后我随便试了 `00`，竟然成功了。后来在 [Setup Amazon AWS EC2 with IPv6 Address](https://xieles.com/blog/setup-amazon-aws-ec2-with-ipv6-address) 看到，这个可以填 00, 01, 02 之类的。

经过这两部配置，可以通过 “Actions, Networking, Manage IP Addresses” 添加 IPv6 address 了，本以为大功告成。但是当我[通过 ping6 验证](https://www.cyberciti.biz/faq/howto-test-ipv6-network-with-ping6-command/)的时候，并没有成功。

```bash
# success
ping6 localhost
# fail
ping6 ipv6.google.com
```

## Update route

这时候才翻出 [Setup Amazon AWS EC2 with IPv6 Address](https://xieles.com/blog/setup-amazon-aws-ec2-with-ipv6-address)，里面说还有设置 route table，在 “Routes >> Edit >> Add another rule >>” 中添加 `::/0`，而 target 设置为跟 ipv4 相同，如下图所示

![](https://xieles.com/wp-content/uploads/2018/06/ipv6fig16.png)

配置完成后，ping6 也成功了！

本来想尝试 [ipv6 over shadowsocks](https://www.polarxiong.com/archives/%E6%90%AD%E5%BB%BAipv6-VPN-%E8%AE%A9ipv4%E4%B8%8Aipv6-%E4%B8%8B%E8%BD%BD%E9%80%9F%E5%BA%A6%E6%8F%90%E5%8D%87%E5%88%B0100M.html)，但是后来发现本地没有 ipv6 网，不过似乎宣称这样可以访问 ipv6 网站。