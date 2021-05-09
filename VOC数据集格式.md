# VOC数据集

## 目录介绍

```shell
VOC_ROOT     #根目录
    ├── JPEGImages         # 存放源图片
    │     ├── aaaa.jpg    
    │     ├── bbbb.jpg  
    │     └── cccc.jpg
    ├── Annotations        # 存放xml文件，与JPEGImages中的图片一一对应，解释图片的内容等等
    │     ├── aaaa.xml
    │     ├── bbbb.xml
    │     └── cccc.xml
    └── ImageSets          
        └── Main
          ├── train.txt    # txt文件中每一行包含一个图片的名称
          └── val.txt
```

其中`JPEGImages`目录中存放的是源图片的数据，(当然图片并不一定要是`.jpg`格式的，只是规定文件夹名字叫`JPEGImages`)；
　　`Annotations`目录中存放的是标注数据，`VOC`的标注是`xml`格式的，文件名与`JPEGImages`中的图片一一对应；
　　`ImageSets/Main`目录中存放的是训练和验证时的文件列表，每行一个文件名(不包含扩展名)，例如`train.txt`是下面这种格式的：

```shell
# train.txt
aaaa
bbbb
cccc
```

## XML标注格式

`xml`格式的标注格式如下：

 ```xml
<annotation>
    <folder>VOC_ROOT</folder>                          
    <filename>aaaa.jpg</filename>  # 文件名
    <size>                         # 图像尺寸（长宽以及通道数）                      
        <width>500</width>
        <height>332</height>
        <depth>3</depth>
    </size>
    <segmented>1</segmented>       # 是否用于分割（在图像物体识别中无所谓）
    <object>                       # 检测到的物体
        <name>horse</name>         # 物体类别
        <pose>Unspecified</pose>   # 拍摄角度，如果是自己的数据集就Unspecified
        <truncated>0</truncated>   # 是否被截断（0表示完整)
        <difficult>0</difficult>   # 目标是否难以识别（0表示容易识别）
        <bndbox>                   # bounding-box（包含左下角和右上角xy坐标）
            <xmin>100</xmin>
            <ymin>96</ymin>
            <xmax>355</xmax>
            <ymax>324</ymax>
        </bndbox>
    </object>
    <object>                       # 检测到多个物体
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>198</xmin>
            <ymin>58</ymin>
            <xmax>286</xmax>
            <ymax>197</ymax>
        </bndbox>
    </object>
</annotation>
 ```

## 制作自己的VOC数据集

制作自己数据集的步骤为：

　　① 新建一个`JPEGImages`的文件夹，把所有图片放到这个目录。(或者使用`ln -s`把图片文件夹软链接到`JPEGImages`)；

　　② 由原来的数据格式生成`xml`，其中`pose`，`truncated`和`difficult`没有指定时使用默认的即可。`bounding box`的格式是`[x1,y1,x2,y2]`，即`[左上角的坐标, 右下角的坐标]`。`x`是宽方向上的，`y`是高方向上的。

　　③ 随机划分训练集和验证集，训练集的文件名列表存放在`ImageSets/Main/train.txt`，验证集的文件名列表存放在`ImageSets/Main/val.txt`。

# COCO数据集

## 目录介绍

```shell
COCO_ROOT     #根目录
    ├── annotations        # 存放json格式的标注
    │     ├── instances_train2017.json  
    │     └── instances_val2017.json
    └── train2017         # 存放图片文件
    │     ├── 000000000001.jpg
    │     ├── 000000000002.jpg
    │     └── 000000000003.jpg
    └── val2017        
          ├── 000000000004.jpg
          └── 000000000005.jpg
```

## json标注格式

与`VOC`一个文件一个`xml`标注不同，`COCO`所有的目标框标注都是放在一个`json`文件中的。这个`json`文件解析出来是一个字典，格式如下：

```json
{
  "info": info,
  "images": [image],
  "annotations": [annotation],
  "categories": [categories],
  "licenses": [license],
}
```

制作自己的数据集的时候`info`和`licenses`是不需要的。只需要中间的三个字段即可。

其中`images`是一个字典的列表，每个图片的格式如下：

```json
# json['images'][0]
{
  'license': 4,
  'file_name': '000000397133.jpg',
  'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
  'height': 427,
  'width': 640,
  'date_captured': '2013-11-14 17:02:52',
  'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
  'id': 397133
}
```

自己的数据集只需要写`file_name`,`height`,`width`和`id`即可。`id`是图片的编号，在`annotations`中也要用到，每张图是唯一的。

`categories`表示所有的类别，格式如下：

```json
[
  {'supercategory': 'person', 'id': 1, 'name': 'person'},
  {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
  {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
  {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
  {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
  {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
  {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
  {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
  {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
  # ....
]
```

`annotations`是检测框的标注，一个bounding box的格式如下：

```json
{
    'segmentation': [[0, 0, 60, 0, 60, 40, 0, 40]],
    'area': 240.000,
    'iscrowd': 0,
    'image_id': 289343,
    'bbox': [0., 0., 60., 40.],
    'category_id': 18,
    'id': 1768
}
```

其中`segmentation`是分割的多边形，如果不知道直接填写`[[x1, y1, x2, y1, x2, y2, x1, y2]]`就可以了，`area`是分割的面积，`bbox`是检测框的`[x, y, w, h]`坐标，`category_id`是类别id，与`categories`中对应,`image_id`图像的id，`id`是`bbox`的`id`，每个检测框是唯一的。

## 一个完整的数据标准格式

自己标注的数据无需info license

```json
{
  "images": [
      { 'file_name': '001.jpg', 'height': 427, 'width': 640, 'id': 1 },        
      { 'file_name': '002.jpg', 'height': 427, 'width': 640, 'id': 2 }
  ],
  "annotations": [
        { 'segmentation': [[0, 0, 60, 0, 60, 40, 0, 40]], 'area': 240.000, 'iscrowd': 0, 'image_id': 289343,  'bbox': [0., 0., 60., 40.], 'category_id': 18, 'id': 1768
        }
  ],
  "categories": [
      {'supercategory': 'person', 'id': 1, 'name': 'person'},
      {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
      {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
      {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
      {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
      {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
      {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
      {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
      {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
  ],
}
```

