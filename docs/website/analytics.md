# Analytics

- [6.2 最全Google Analytics(分析)使用教程 - 知乎](https://zhuanlan.zhihu.com/p/134682010)

## Upgrade to GA4

Currently, I am using `analytics.js`, but GA4 requires `gtag.js`. As a result, the [setup wizard](https://support.google.com/analytics/answer/9744165#zippy=%2Cadd-your-tag-directly-to-your-web-pages) does not successed.

![](https://user-images.githubusercontent.com/13688320/130027103-7ad54be7-2e57-4137-9d47-73dbe3119fd0.png)

But anyway, we still need to `Create Property` instead of skipping this step.

After creating the property, we will have two properties for `ESL`, 

![image](https://user-images.githubusercontent.com/13688320/130027600-0c73aa2c-4d2c-40e8-8b22-36ef5e34ace4.png)

But we have not added the javascript for ESL-GA4. The tracking code can be found via

`Admin` -> `ESL-GA4` -> `Data Streams` -> `Web` -> `Add new on-page tag` 

and then paste to the source of the webpages. Since `ESL-GA4` is another property from `ESL`, do not remove the existing tracking code for `ESL`.

**(Don't remove the old analytics.js tag; it will continue to collect data for your Universal Analytics property. The gtag.js tag that you're adding will collect data for your new Google Analytics 4 property.)**

Refer to [向已设置 Analytics 的网站添加 Google Analytics（分析）4 媒体资源 - Google Analytics（分析）帮助](https://support.google.com/analytics/answer/9744165)

Blogs on explaining the difference between different versions of Google Analytics (GA)

- [新版谷歌分析 GA4 详细设置/更新指南](https://zhuanlan.zhihu.com/p/369419998)