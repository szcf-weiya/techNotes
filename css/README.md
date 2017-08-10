## 利用@media screen实现网页布局的自适应

参考http://www.cnblogs.com/xcxc/p/4531846.html

### 1280分辨率以上（大于1200px）

```
@media screen and (min-width:1200px){
}
```

### 1100分辨率（大于960px，小于1199px）
```
@media screen and (min-width: 960px) and (max-width: 1199px) {

}
```

### 880分辨率（大于768px，小于959px）

```
@media screen and (min-width: 768px) and (max-width: 959px) {

}
```
