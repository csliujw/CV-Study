[知乎简介](https://zhuanlan.zhihu.com/p/337479181)

[论文地址](https://arxiv.org/pdf/2011.09763.pdf)

# 摘要

生物医学图像分割，细胞分割是瓶颈。

Attention-based transformers are state-of-the-art in a range of deep learning fields. 基于注意力的transformers在深度学习领域效果表现很好，`SOTA`。

将注意力机制加入分割中，效果很好，比其他方法更优秀。

因此，本文提出了一种基于注意力机制的`Cell DETR（细胞变换检测器）`，可进行端到端的实例分割，在特定数据下，分割效果与`MaskRCNN`相当，且速度更快。

| name       | iou  | time   |
| ---------- | ---- | ------ |
| Mask R-CNN | 0.84 | 29.8ms |
| Cell-DETRA | 0.83 | 9.0ms  |
| Cell-DETRB | 0.84 | 21.2ms |

`思路来源：`N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” arXiv:2005.12872, 2020.  提出了一种新颖的基于注意力的transform DETR用于全景分割，这个方法简单且高效。可以考虑实际应用。

# 简介

Instance segmentation is a major bottleneck in quantifying single cell microscopy data and manual analysis is prohibitively labour intensive （实例分割是量化单细胞显微镜数据的主要瓶颈，手工分析过于费力。）

注意力机制方法，如最近提出的detection transformer DETR，正日益超过其他方法。

- 本文提出了Cell-DETR（a novel attention-based detection transformer for instance segmentation of biomedical samples based on DETR ），一种用于生物医学样本实例分割的基于注意力的detection transformer

- 解决了分割瓶颈
    - 我们解决了微结构化环境中酵母细胞的自动细胞实例分割瓶颈（图1），

![image-20210412175739872](..\..\pics\CV\ISG\Untitled\image-20210412175739872.png)

- 介绍以前的分割方法
- 介绍显微镜实验装置、测试过的架构以及train和evaluation方法在
- 分析方法性能，与实例分割baseline进行对比，超越了之前的方法（`fps或iou精度`），可进行在线检测。

# 背景

Background

UNet效果好！为了实例分割要对分割图进行额外的处理！NLP里，基于注意力机制的方法在CV中大放异彩！最近，一个transformer-based method（DETR）用于物体检测和全景分割可以与Faster R-CNN媲美，这给了我们新的希望，可以进一步改进自动目标检测和分割性能（automated object detection and segmentation performance`）【我觉得更多的是性能上的改进】`

减少了细胞观察的环境，怎么处理的，杂七杂八的。

# 方法

需要分割的细胞周围会有一些trap细胞！！

![image-20210412203348924](..\..\pics\CV\ISG\Untitled\image-20210412203348924.png)

带背景（浅灰色）的注释用于语义分段训练，例如U-Net。 对实例分割训练我们引入了无对象类∅来代替背景类。

----

来自各种实验的带注释的419个样本图像集被随机分配用于网络训练，验证和测试（分别为76％，12％和12％）

trap instances in shades of dark grey, cell instances in shades of violet and transparent background; 

trap实例细胞用灰色，细胞实例用紫色（violet）

> Cell-DETR实例分割架构

提出了基于DETR全景分割架构的Cell-DETR模型A和模型B

DETR模型与Cell-DETR A/B模型的区别：

![image-20210412205421479](..\..\pics\CV\ISG\Untitled\image-20210412205421479.png)

Cell-DETR变体的参数比原始参数大约少一个数量级。

> Cell-DETR模型

They are the backbone CNN encoder, the transformer encoder-decoder, the bounding box and class prediction heads, and the segmentation head.

主干（backbone）是CNN编码其，transformer编码解码其，边界框和class prediction heads， segmentation head。【They are the backbone CNN encoder, the transformer encoder-decoder, the bounding box and class prediction heads, and the segmentation head.】

- CNN编码器提取图像特征。CNN基于具有64、128、256和256个卷积滤波器的四个类似ResNet块。在每个块之后，将使用2×2平均池化层对中间特征图进行下采样。 Cell-DETR变体采用不同的激活和卷积，如表I所示。

![image-20210412210336781](..\..\pics\CV\ISG\Untitled\image-20210412210336781.png)