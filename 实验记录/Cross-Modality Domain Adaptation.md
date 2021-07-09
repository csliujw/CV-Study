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
