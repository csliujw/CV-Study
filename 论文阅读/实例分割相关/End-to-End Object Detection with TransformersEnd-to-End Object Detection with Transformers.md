# 博客解读

## 创新点

将目标检测任务转化为一个序列预测（set prediction）的任务，使用transformer编码-解码器结构和双边匹配的方法，由输入图像直接得到预测结果序列。和SOTA的检测方法不同，没有proposal（Faster R-CNN），没有anchor（YOLO），没有center(CenterNet)，也没有繁琐的NMS，直接预测检测框和类别，利用二分图匹配的匈牙利算法，将CNN和transformer巧妙的结合，实现目标检测的任务。

![](https://pic1.zhimg.com/v2-477a4e2a04b4913e1d8dd4b67e4df0f0_r.jpg)

在本文的检测框架中，有两个至关重要的因素：

①使预测框和ground truth之间一对一匹配的序列预测loss；

②预测一组目标序列，并对它们之间关系进行建模的网络结构。接下来依次介绍这两个因素的设计方法。

## 模型的整体结构

![](https://pic1.zhimg.com/80/v2-aae1329060cd9d50df17c4e7a421e09c_720w.jpg)

Backbone + transformer + Prediction

CNN + encoder+decoder + FFN

> Backbone

利用传统的CNN网络，将输入的图像 ![[公式]](https://www.zhihu.com/equation?tex=3+%5Ctimes+W_%7B0%7D+%5Ctimes+H_%7B0%7D) 变成尺度为 ![[公式]](https://www.zhihu.com/equation?tex=2048+%5Ctimes++%5Cfrac%7BW_%7B0%7D%7D%7B32%7D+%5Ctimes+%5Cfrac%7BH_%7B0%7D%7D%7B32%7D) 的特征图

> transformer

![](https://pic2.zhimg.com/v2-1be61511d53dca07f1c83697eb23a87d_r.jpg)

**Transformer encoder**部分首先将输入的特征图降维并flatten，然后送入下图左半部分所示的结构中，和空间位置编码一起并行经过多个自注意力分支、正则化和FFN，得到一组长度为N的预测目标序列。其中，每个自注意力分支的工作原理为可参考[刘岩：详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)，也可以参照论文：[https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

接着，将Transformer encoder得到的预测目标序列经过上图右半部分所示的Transformer decoder，并行的解码得到输出序列（而不是像机器翻译那样逐个元素输出）。和传统的autogreesive机制不同，每个层可以解码N个目标，由于解码器的位置不变性，即调换输入顺序结果不变，除了每个像素本身的信息，位置信息也很重要，所以这N个输入嵌入必须不同以产生不同的结果，所以学习NLP里面的方法，加入positional encoding并且每层都加，**作者非常用力的在处理position的问题，在使用 transformer 处理图片类的输入的时候，一定要注意position的问题。**

>预测头部（FFN）

使用共享参数的FFNs（由一个具有ReLU激活函数和d维隐藏层的3层感知器和一个线性投影层构成）独立解码为包含类别得分和预测框坐标的最终检测结果（N个），FFN预测框的标准化中心坐标，高度和宽度w.r.t. 输入图像，然后线性层使用softmax函数预测类标签。

## 模型损失函数

基于序列预测的思想，作者将网络的预测结果看作一个长度为N的固定顺序序列 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D%3D%7B%5Ctilde%7By%7D_%7Bi%7D%7D%2C%5C+i%5Cepsilon%281%2CN%29) ,（其中N值固定，且远大于图中ground truth目标的数量） ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Ctilde%7By%7D_%7Bi%7D%7D%3D%28%5Ctilde%7Bc_%7Bi%7D%7D%2C%5Ctilde%7Bb%7D_%7Bi%7D%29) ，同时将ground truth也看作一个序列 ![[公式]](https://www.zhihu.com/equation?tex=y%3Ay_%7Bi%7D%3D%28c_%7Bi%7D%2Cb_%7Bi%7D%29) （长度一定不足N，所以用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) （表示无对象）对该序列进行填充，可理解为背景类别，使其长度等于N），其中 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bi%7D) 表示该目标所属真实类别， ![[公式]](https://www.zhihu.com/equation?tex=b_%7Bi%7D) 表示为一个四元组（含目标框的中心点坐标和宽高，且均为相对图像的比例坐标）。

那么预测任务就可以看作是 ![[公式]](https://www.zhihu.com/equation?tex=y%E4%B8%8E%5Ctilde%7By%7D) 之间的二分图匹配问题，采用匈牙利算法[[1\]](https://zhuanlan.zhihu.com/p/144974069#ref_1)作为二分匹配算法的求解方法，定义最小匹配的策略如下：

![img](https://pic1.zhimg.com/80/v2-8e3856e0d4bf2f3feb44e032bab5f7e0_720w.jpg)

求出最小损失时的匹配策略 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Csigma%7D) ，对于 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmatch%7D) 同时考虑了类别预测损失即真实框之间的相似度预测。

对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28i%29) , ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bi%7D) 的预测类别置信度为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BP%7D_%7B%5Csigma%28i%29%7D%28c_%7Bi%7D%29) ,边界框预测为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bb%7D_%7B%5Csigma%28i%29%7D) ,对于非空的匹配，定于 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmatch%7D) 为：

![img](https://pic4.zhimg.com/80/v2-67ba49e2b04107cf3507090c062374ef_720w.png)

进而得出整体的损失：

![img](https://pic1.zhimg.com/80/v2-fbf5d4eff6d770f23f2bbc46278ec890_720w.png)

考虑到尺度的问题，将L1损失和iou损失线性组合，得出![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D)如下所示：

![img](https://pic3.zhimg.com/80/v2-e3105edd50e92ae3e7cb9adb3ad69926_720w.png)

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D) 采用的是Generalized intersection over union论文提出的GIOU[[2\]](https://zhuanlan.zhihu.com/p/144974069#ref_2),关于GIOU后面会大致介绍。

为了展示DETR的扩展应用能力，作者还简单设计了一个基于DETR的全景分割框架，结构如下：

![preview](https://pic1.zhimg.com/v2-2b9fad8f3430b22f47251fd62394f108_r.jpg)

## 结果

其中DETR对于大目标的检测效果有所提升，但在小目标的检测中表现较差。

# 摘要

提出了一种新的方法，将目标检测视为直接集预测问题（将目标检测作为预测一个集合的问题？）。我们的方法简化（streamlines）了检测的流程，有效地消除了对许多手动设计组件的需求，如非最大抑制过程或锚生成，这些组件需要我们对预测任务有一定的常识性经验（先验知识）。名为`DEtection TRansformer`或者叫`DETR`的新框架组成部分（ingredients）是一种基于集合的全局损失，它通过`二分匹配`和`transformer encoder-decoder`结构来强制进行唯一的预测。`【好像是用到了匈牙利算法】`。给定一个固定的小的学习对象查询集合，`DETR`对对象和全局图像上下文的关系进行推理，以直接并行输出最终的预测集。与许多其他现代探测器不同，新模型在概念上很简单，不需要专门的库。在coco目标检测数据集上，DETR展现了与Faster R-CNN baseline相当的准确性和高效性。此外，DETR可以很容易地推广，以统一的方式产生全景分割。我们证实了，它明显优于竞争基线。

----

Abstract. We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr.

# 引言