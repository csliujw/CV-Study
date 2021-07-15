# Cross-Modality Domain Adaptation for Medical Image Segmentation简介

## 针对问题

第一个无监督交叉模态域自适应方法的医学成像基准（从对比增强的图像T1 到高分辨率图像T2）

用增强图像进行训练，预测原始的高清图像。

## 动机

Domain Adaptation，通过鼓励算法对不可见的情况或不同输入数据领域具有鲁棒性。已有了一些图像分割的DA技术，但是他们的数据集小，且主要处单类问题。==该挑战是无监督跨模态领域自适应大型多类数据集。==

## 任务

该挑战的目标是分割涉及前庭神经鞘瘤(VS)随访和治疗计划的两个关键大脑结构:肿瘤和耳蜗。对比增强T1 (ceT1)磁共振成像(MRI)扫描通常用于前庭神经鞘瘤（VS）分割，最近的工作表明高分辨率T2 (hrT2)成像可能是一种可靠、更安全、成本更低的替代ceT1。基于这些原因，我们提出一种==unsupervised cross-modality challenge==(从ceT1到hrT2)，目的是在hrT2扫描中自动执行VS和耳蜗分割。训练源和目标集分别为未配对标注的ceT1扫描和未标注的hrT2扫描。

## 评估指标

Mean Dice Score

## 规则

可以使用的有：

- 空间归一化MNI空间
- 使用经典的基于单图谱的工具（例如，SPM）

## 数据开放

- 训练集和验证集。
- 测试集不开放给用户。

# 数据特点

All images were obtained on a 32-channel Siemens Avanto 1.5T scanner using a Siemens single-channel head coil:

- Contrast-enhanced T1-weighted imaging was performed with an MPRAGE sequence with in-plane resolution of 0.4×0.4 mm, in-plane matrix of 512×512, and slice thickness of 1.0 to 1.5 mm (TR=1900 ms, TE=2.97 ms, TI=1100 ms)
- High-resolution T2-weighted imaging was performed with a 3D CISS or FIESTA sequence in-plane resolution of 0.5x0.5 mm, in-plane matrix of 384x384 or 448x448, and slice thickness of 1.0 to 1.5 mm (TR=9.4 ms, TE=4.23ms).

MORAGE：three-dimensional三维图

# 准备工作

## 环境安装

> 读取特定格式的医疗图像数据nii.gz数据

- 安装SimpleITK 
  - annconda官网搜索SimpleITK，下载，conda离线安装
  - <a href="https://anaconda.org/SimpleITK/SimpleITK/files?__cf_chl_jschl_tk__=368889a7788297669d509b1013d56375bcc4322f-1623317860-0-AXD0oBk6I8EKANYg9LgQg06SCa5sxaFwY9XYqNm1YiMLZjpGinKSBxIaLD0Y4qJAvOxBO8zQIWR-sdZDh6FX9K82QYwM3acvhZCMMnbcfaFjoJGuM1KUcM9jcRD8EI8rlKUiLIlR2M3tbwcMmYbvQXHBqZyJAJmFezmHnbpvIfdci9WkOawc4pZn0GXgDseVPHbtwo4s3950ExCga4Van2m_gZ5UT160S3wyOvmjOuvFKqpRevSOc0ryYKWaulXZPTahR8lieFA7JrMiTBAdSybLfAvQ1kcxUKZrMq7WARkY8C-3ADai6nZxfWMs2g4XsKvVGITfUWQz0KOAdiqcV7ploK1K3dEDTC1Vagd31aV6Kcevxot0fIbkRBfZ7xIdek4iw9AMQvmq-HCZWJAwkLDUht1fO9aBRtbqrOGrcXg97N9vT7ueoYSDRC57E56oLltxmUoCyqD4J9NqHkt07yrWk_q9mDvgaCnwAOsVaCck">网址</a>
- 安装nibabel `pip install nibabel`

> 读取图片代码

```python
import os

import SimpleITK as sitk
import nibabel as nib
from matplotlib import pylab as plt
from matplotlib import pyplot as plt

source_training = "./training/source/"
target_training = "./training/target/"
target_validation = "./validation/"

srcs = [source_training, target_training, target_validation]


def showNii_pic(src):
    itk_img = sitk.ReadImage()
    # 获得图像内容
    img = sitk.GetArrayFromImage(itk_img)
    print(img.shape)  # (120, 512, 512)表示各个维度的切片数量

    for i in range(img.shape[0]):
        plt.imshow(img[i, :, :], cmap='gray')
        plt.show()


def nib_plt(src):
    img = nib.load(src)
    # 由文件本身维度确定，可能是3维，也可能是4维
    width, height, queue = img.dataobj.shape
    return width, height, queue


def show_channels(src):
    paths = os.listdir(src)
    paths = [os.path.join(src, p) for p in paths]
    d = set()
    for i in paths:
        _, _, c = nib_plt(i)
        d.add(c)
    print(d)


def show_single_channels(src):
    _, _, c = nib_plt(src)
    return c


if __name__ == '__main__':
    train_dir = 'F:/crossModa/crossmoda_training/source_training/'
    target_dir = 'F:/crossModa/crossmoda_training/target_training/'
    val_dir = 'F:/crossModa/crossmoda_validation/target_validation/'

    ceT1 = set()
    for i in range(1, 105):
        c = show_single_channels(train_dir + f'crossmoda_{i}_ceT1.nii.gz')
        ceT1.add(c)
    print("=" * 50)

    Label = set()
    for i in range(1, 105):
        c = show_single_channels(train_dir + f'crossmoda_{i}_Label.nii.gz')
        Label.add(c)
    print("=" * 50)

    target = set()
    for i in range(106, 210):
        c = show_single_channels(target_dir + f'crossmoda_{i}_hrT2.nii.gz')
        target.add(c)
    print("=" * 50)

    val = set()
    for i in range(211, 242):
        c = show_single_channels(val_dir + f'crossmoda_{i}_hrT2.nii.gz')
        val.add(c)

    print(f"ceT1:{ceT1} ==== Label:{Label} ==== target:{target} ==== val:{val}")
# ceT1:{120, 160} ==== Label:{120, 160} ==== target:{80, 40, 20, 70} ==== val:{80, 40}
```

## 数据特点

三维数据，数据的维度不太一样

```shell
--- training
	--- source，对比增强的数据T1和其标签。 
		--- ceT1: 120、160通道 图片大小512*512
		--- Label：120、160通道 图片大小512*512
	--- target，高清晰度的数据T2。
		--- 数据是20、40、70、80通道 图片大小 448*448, 384*384
--- validation，高清晰度的数据T2。
	验证数据集 40、80通道 图片大小 448*448, 384*384
```

# 选用的论文与代码

> 论文

Squeeze-and-Excitation Normalization for Automated Delineation of Head and Neck Primary Tumors in Combined PET and CT Images

> 代码

https://github.com/iantsen/hecktor

## 代码修改

先调整读数据集那块。

我发现，train的时候只使用了input与mask。故先处理这两个试试

# 领域自适应

[【TL学习笔记】1：领域自适应(Domain Adaptation)方法综述_LauZyHou的笔记-CSDN博客](https://blog.csdn.net/SHU15121856/article/details/106874558)

把分布不同的源域和目标域的数据，映射到一个特征空间中，使其在该空间中的距离尽可能近。于是在特征空间中对source domain训练的目标函数，就可以迁移到target domain上，提高target domain上的准确率。

通过减小源域（辅助领域）到目标域的分布差异，进行知识迁移，从而实现数据标定。

> 核心思想

- 找到不同任务之间的相关性。
- 举一反三、照猫画虎。

## 为什么要迁移学习

- 数据角度
  - 手机数据困难
  - 为数据打标签耗时
  - 训练一对一模型很繁琐

- 模型的角度
  - 个性化模块很复杂
  - 云+端的模型需要作具体化适配
- 应用角度
  - 冷启动问题：没有足够用户数据，推荐系统无法工作。

## 常见迁移学习分类

> 传统机器学习方法是最小化损失

$$
min \frac{1}{n} \sum^n_{i=1}  L(x_i,y_i,\theta)
$$



> 基于实例的迁移

是考虑到源域中总有一些样本和目标域样本很相似，那么就将源域的所有样本的Loss在训练时都乘以一个权重 $w_i$ 表示看重程度，和目标域越相似的样本，权重越大。
$$
min \frac{1}{n} \sum^n_{i=1}  w_iL(x^s_i,y^s_i,\theta)
$$
 

> 基于特征的迁移 ★

是将源域样本和目标域样本用一个映射Φ调整到同一个特征空间，这样在这个特征空间样本能够“对齐”，这也是最常用的方法：
$$
min \frac{1}{n} \sum^n_{i=1} L(Φ(x^s_i),y^s_i,\theta)
$$


> 基于模型的迁移★

利用源域和目标域的参数共享模型;是找到新的参数$θ ′ $，通过参数的迁移使得模型能更好的在目标域上工作：
$$
min \frac{1}{n} \sum^n_{i=1} L(x^s_i,y^s_i,θ ′ )
$$


> 基于关系的迁移【用的少】

如果目标域数据没有标签，就没法用Fine-Tune把目标域数据扔进去训练，这时候无监督的自适应方法就是基于特征的自适应。因为有很多能衡量源域和目标域数据的距离的数学公式，那么就能把距离计算出来嵌入到网络中作为Loss来训练，这样就能优化让这个距离逐渐变小，最终训练出来的模型就将源域和目标域就被放在一个足够近的特征空间里了。

这些衡量源域和目标域数据距离的数学公式有KL Divergence、MMD、H-divergence和Wasserstein distance等。



利用源域中的逻辑网络关系进行迁移

-----

<img src="..\pics\CV\AD\image-20210713103102382.png">

同构：特征维度一样（领域自适应）

异构：特征维度不一样，如图像于文本

## 领域自适应基本概念

### 基本概念

- 域（Domain）：由数据特征和特征分布组成，是学习的主题
  - Source domain（源域）：已有知识的域
  - Tarrget domain（目标域）：要进行学习的域
- 任务（Task）：由目标函数和学习结果组成，是学习的结果

### 形式化

- 条件：给定一个源域$D_s$和源域上的学习任务$T_s$，目标域 $D_t$ 和 目标域上的学习任务$T_T$
- 目标：利用 $D_s$ 和 $T_s$ 学习在目标域上的预测函数 $f(·)$
- 限制条件：$D_s ≠ D_T$ 或 $T_S ≠ T_T$  【两个域不一样或两个任务不一样】

## 领域自适应问题

- Domain Adaptation；cross-domain learning；同构迁移学习
- 问题定义：有标签的源域和无标签的目标域共享相同的特征和类别，但是特征分布不同，如何利用源域标定目标域 $D_s≠ D_T：P_S(X) ≠ P_T(X)$ ， 即源域和目标域的边缘分布不一样
- CVPR、ICCV、ICML、NIPS、IJCAI、AAAI 均有相关论文

----

> 按照目标域有无标签

- 目标域全部有标签：supervised DA
- 目标域有一些标签：semi-supervised  DA
- 目标域全没有标签：unsupervised DA  ★【关注点】

### 方法概览

> 基本假设

数据分布角度：源域和目标域的概率分布相似

- 最小化概率分布距离

特征选择角度：源域和目标域共享着某些特征

- 选择出这部分公共特征

特征变换角度：源域和目标域共享某些子空间

- 把两个域变换到相同的子空间

> 解决思路

数据分布：概率分布适配法

特征选择：特征选择法

特征变换：子空间学习法

## 深度学习中的领域自适应方法

