# 项目名称
基于飞浆的paddleX套件实现红绿灯的检测与识别

# 目录
* 项目介绍
* VOC数据集介绍
* 环境搭建
* 构建数据集
* 使用paddleX进行模型训练

# 项目介绍
该项目的主要目的是使用paddlex套件完成红绿灯的检测和识别，训练数据集和验证数据集均使用飞浆的公开数据集`ccshldzs`。
## paddleX套件介绍
了解深度学习的小伙伴都知道当我们需要落地一个算法的时候，大致的流程一般都是：`收集数据 -> 清洗数据 -> 模型训练 -> 模型部署`。

为了简化流程和减少重复代码的编写，飞浆推出了PaddleX套件。PaddleX尽可能将上述流程整合到一起，使得开发人员只需要编写较少的代码就可以完成一个模型的训练和部署，大大简化了开发人员的工作。

目前使用PaddleX可以完成常见的深度学习任务，如分类、检测和分割。同时，paddleX还内置了常见的model和预训练权重，我们只需要使用较小的数据,在预训练权重的基础上继续训练就可以获得不错的效果,大大提高了开发效率。

## 项目模型介绍
本次项目是使用paddleX内置的yolov3模型完成红绿灯的检测，Yolo是一个主流的一阶段目标检测算法，其检测速度快，常用在需要实时检测的场景中。

数据集使用的是公开数据集`ccshldzs`，数据集格式为`VOC格式`。

# VOC数据集介绍
VOC 格式是一种常见的数据集格式，整体理解起来就是：
* 所有的图片都放在一个文件夹下
* 每张图片都至少对应着一个标注文件，标注文件的格式为xml。因为VOC格式可以用来检测和分割，所以一张图片可以和多个xml标注文件对应。
* 为了标明每张图片与xml标注文件的对应关系，还需要额外有一个train.txt文件存储该对应关系。

下面是VOC官方数据的格式，大家可以参考一下，[参考博客](https://blog.csdn.net/qq_41289920/article/details/105940011)

```
.
├── Annotations 进行 detection 任务时的标签文件，xml 形式，文件名与图片名一一对应
├── ImageSets 包含三个子文件夹 Layout、Main、Segmentation，其中 Main 存放的是分类和检测的数据集分割文件
├── JPEGImages 存放 .jpg 格式的图片文件
├── SegmentationClass 存放按照 class 分割的图片
└── SegmentationObject 存放按照 object 分割的图片
 
├── Main
│   ├── train.txt 写着用于训练的图片名称， 共 2501 个
│   ├── val.txt 写着用于验证的图片名称，共 2510 个
│   ├── trainval.txt train与val的合集。共 5011 个
│   ├── test.txt 写着用于测试的图片名称，共 4952 个
```

# 环境搭建
## 环境介绍
本次项目我们使用最新的paddle 2.2.2版本, paddleX也使用2.1.x版本。

由于paddle是系统自带的，所以我们只需要安装paddleX即可。

## 安装paddleX
执行如下代码安装paddleX
```
# 安装paddlex
!pip install paddlex
```
# 构建数据集
## 解压数据集文件到指定位置
执行如下代码将数据集解压到work文件夹下，该文件夹下的文件重启后不会消失，会一直保存。

```
# 解压数据集到相应位置
!unzip data/data35732/ccchldzs.zip -d work/
```

## 切换工作路径
aistudio默认的初始工作路径为`/home/aistudio`，我们先将工作路径切换到`/home/aistudio/work`下，方便后续的操作。

```
# 切换工作路径
import os
os.chdir('/home/aistudio/work')
```

## 开启数据增强
为了增加模型的鲁棒性，我们一般会对原始图像做一些数据增强操作，如翻转、裁剪，旋转，填充、调整亮度等，以此来增加图像的复杂性，提升模型的鲁棒性。

`paddleX.transform`为我们提供了常用的数据增强操作。在使用时，既可以逐个使用某个数据增强操作处理一张图片，也可以将多个数据增强操作放入一个`paddleX.transforms.Compose`类中，那么将会依次使用数据增强操作处理每一张图片。

## 定义Dataset

在常见的深度学习框架中我们如果要定义自己的dataset，一般会首先继承框架的某个基类，如`paddle`框架中的`paddle.io.Dataset`类，重写基类中的某些方法，来自定义Dataset。

但`paddleX`已经为我们定义好了常见数据集格式的Dataset类，我们只需要指定数据集所在的文件即可生成对应数据集格式的Dataset。如`paddlex.datasets.VOCDetection`对应VOC格式的数据，`paddlex.datasets.CocoDetection`对应COCO格式的数据，`paddlex.datasets.SegDataset`对应分割格式的数据。

我们本次使用的数据集格式为VOC格式，所以直接使用`paddlex.datasets.VOCDetection`类加载数据集就可以。

* data_dir: 该参数指定训练图像所在的文件夹，该文件夹如图1所示
* file_list: 该参数指定包含所有训练图像名称及其对应标注文件名称的文件，如图2所示
* label_list: 该参数指定包含所有标签id与名称对应关系的文件，文件内容如图3所示
* transforms: 该参数指定数据增强的序列组合
* shuffle: 该参数指定数据加载是否随机

图一
![图一](attachment:c3e077e2-1102-406a-a61d-d84dab393e22.png)


图二
![图二](attachment:f45f6edb-738e-4855-b9e7-161ee950ed23.png)


图三
![image.png](attachment:fab1a8fb-99cb-4015-aa28-9b34f7d7ebad.png)

具体代码如下:

```
import paddlex as pdx
from paddlex import transforms

# 数据增强
train_transforms = transforms.Compose([
    transforms.RandomExpand(),                                 #随即填充
    transforms.RandomHorizontalFlip(),                         #随机水平翻转
    transforms.MixupImage(mixup_epoch=250),                    #Mixup策略
    transforms.RandomCrop(),                                   #随机裁剪
    transforms.Resize(target_size=608, interp='RANDOM'),       #随机调整图像大小
    transforms.Normalize(),                                    #图像标准化（均值除以标准差）
    # transforms.RandomDistort(brightness_range=0.5, brightness_prob=0, contrast_range=0.5, contrast_prob=0, saturation_range=0.5, saturation_prob=0, hue_range=18, hue_prob=0),    
    #随机 prob是概率，置零关掉             亮度 默认0.5 0.5                        对比度   0.5 0.5                   饱和度 0.5 0.5                  色调 18 0.5
    # transforms.ResizeByShort(short_size=800, max_size=1333),   #根据图像的短边调整图像大小        客户端上没有
    # transforms.Padding(coarsest_stride=1)                     #将图像的长和宽padding至32的倍数   客户端上没有
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])

# 定义数据集
train_dataset = pdx.datasets.VOCDetection(
    data_dir='ccchldzs/train',
    file_list='ccchldzs/train/train_list.txt',
    label_list='ccchldzs/train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='ccchldzs/dev',
    file_list='ccchldzs/dev/val_list.txt',
    label_list='ccchldzs/dev/labels.txt',
    transforms=eval_transforms)
```

# 模型训练
检测红绿灯是一个目标检测任务，所以本次我们使用paddleX自带的`yolov3`模型进行训练。

## 步骤一：创建model
这一步就是简单的实例化`paddlex.det.YOLOv3`类即可

## 步骤二：指定训练参数
创建好model后，可以直接调用train函数进行训练和验证模型的准确度，下面是`train`函数的参数解释

* num_epochs: 训练的轮次
* train_dataset: 训练时的dataset
* train_batch_size: 训练时的batch_size大小
* eval_dataset: 验证时的dataset
* learning_rate: 模型初始学习率
* warmup_steps: 
* warmup_start_lr:
* save_interval_epochs: 训练几个epoch保存一次权重
* lr_decay_epochs: 指定调整学习率的轮次数
* save_dir: 指定权重和日志保存的文件夹

```
# 使用yolov3开启训练
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop//docs/apis/models/detection.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/develop//docs/parameters.md
model.train(
    num_epochs=1,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.0001,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=1,
    # lr_decay_epochs=[],
    save_dir='output/yolov3_darknet53')
```

# 结果分析
本次训练我只训练了一个epoch，可以看到最终的准确度在90左右，还是不错的，大家可以多训练几轮看看效果如何。

最终的权重保存在了`output/yolov3_darknet53/epoch_1`中。


# 可视化验证效果

```
import paddlex as pdx
import matplotlib.pyplot as plt
import cv2
import numpy as np

%matplotlib inline

test_jpg = 'work/ccchldzs/test/20002.jpg'
model = pdx.load_model('work/output/yolov3_darknet53/best_model')

# predict接口并未过滤低置信度识别结果，用户根据需求按score值进行过滤
result = model.predict(test_jpg)

# 可视化结果存储在./visualized_test.jpg, 见下图
pdx.det.visualize(test_jpg, result, threshold=0.3, save_dir='./')

image = cv2.imread('visualize_20002.jpg')
image = image[:, :, ::-1]
# 展示最终结果
plt.imshow(image)
plt.show()
```

# 学习感悟

不得不说，paddle总体使用起来还是很好的，官方文档写的很详细，从pytorch转过来也比较容易，给开发的工程师大牛们点个赞，希望自己以后也能有机会到大的平台，加油加油！

# 项目链接
aistudio链接：https://aistudio.baidu.com/aistudio/projectdetail/3531266

github链接：https://github.com/Tomoko-hjf/-AI-/edit/main/README.md



















