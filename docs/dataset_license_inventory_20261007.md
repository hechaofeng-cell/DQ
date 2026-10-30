# 数据来源、许可与发布范围

此清单用于本地投稿材料整理，不代表已完成所有源图的再分发许可验证。便携复算包提供目标级派生状态和冻结预测，不打包源图片集合或训练权重。论文图示包含四张COCO源图的衍生展示，正式发布前应核对对应图片权利记录。

| 来源 | 官方记录 | 当前可确认内容 | 发布范围处理 |
|---|---|---|---|
| COCO | [官方使用条款](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm) | 标注按CC BY 4.0；COCO不拥有图片版权，图片依照各Flickr来源条款 | 保留image_id、框与来源；不以标注许可推断全体图片可自由再分发 |
| Open Images V7 | [官方许可说明](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses) | 标注CC BY 4.0；图片记录为CC BY 2.0，同时官方要求自行核实图片许可有效性 | 按image_id回查原作者、来源与实际许可，保留署名 |
| Objects365 | [官方下载说明](https://www.objects365.org/download.html) | 学术研究用途；标注CC BY 4.0；官方不拥有源图版权，并要求不要再分发图片 | 便携包不提供Objects365源图；保留原ID与下载说明 |
| PASCAL VOC | [官方数据资源](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) | 官方数据和标注用于研究；没有在本轮建立全部源图的统一再分发许可 | 保留VOC年份和image_id；投稿发布前按来源检查具体条款 |

## 正文图6的图示来源

四个COCO train2017 image_id分别是036451、111543、087444、018809。目标框、源图与裁剪图SHA256、缓存预测和确定性示例选择规则见case_manifest.json。该记录追溯图示来源，不包含已核验的摄影者署名或逐图许可，因此不能据此声明图示发布权利已经完结。

## 待作者确定的事项

作者姓名、单位、邮箱、资助、分工和利益冲突依用户要求暂留待填。公共材料持久访问地址尚未建立；当前ZIP是本地审阅与复算材料。没有上传到公共应用或执行投稿。
