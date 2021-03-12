# Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth


# 简要描述

Abstract
- 当前最先进实例分割方法实时性不高，不适用于自动驾驶这些实时应用。虽然精度高但是fps太低，且生成的mask分辨率低。
- 提出了一种新的高精度的 速度快的实例分割模型，生成的mask分辨率高，在200w像素的图像上速度超过10帧每秒。


Introduction
- Figure1 聚类算法恢复实例
- 当前方法基于检测+分割。使用边界框检测方法检测对象，然后为每个对象生成二进制掩码。
- 迄今为止，Mask RCNN依旧是最常用，且效果出色的框架。MaskRCNN在精度方面提供了良好的结果，但是生成的低分辨率mask不总是让人满意。生成mask速度也慢，不适用于自动驾驶这种实时应用。

- 实例分割有一个流行的分支==> proposal-free methods

2.标记像素，然后聚类

对图像的每个像素进行分类标记。接下来是使用聚类算法将像素分组到对象实例中。下图显示了一般框架。



该方法受益于语义分割，可以预测高分辨率的对象掩模。与分割检测跟踪技术相比，标签像素跟踪聚类方法在经常使用的基准上精度较低。由于像素标记需要密集的计算，通常需要更多的计算能力。
https://blog.csdn.net/Yong_Qi2015/article/details/107777080



# 详细翻译

## Abstract

​		目前，最先进的实例分割方法并不适用于像自动驾驶这样的实时应用程序，这些实时应用程序需要快速的执行速度和很高的精度。虽然目前proposal-based（基于提案）的方法具有很高的精度，但是它们生成mask的速度慢且生成的mask结果分辨率低。相比之下，proposal-free方法可以快速生成高分辨率的mask，但是精度不如proposal-based。In this work 我们为proposal-free实例分割提出了一种新的聚类损失函数。这个损失函数将属于同一实例像素的spatial embeddings（嵌入空间）拉到了一起，共同学习特定实例的clustering bandwidth（聚类的带宽），以便最大化实例结果mask的iou。当和一个快速的结构相结合时，网络可以实时的进行实例分割且保持一个较高的精度。我们在具有挑战性的Cityscapes基准上评估了我们的方法，并在200万像素的图像上以超过10fps/s的速度获得了最佳的效果。（超过Mask RCNN 5%）

## Introduction

​		语义实例分割的任务是定位图片中的所有目标对象，并为每个对象赋值一个特殊的类别并为每个对象生成a pixel-perfect mask，完美勾勒出他们的形状。这个标准的边框目标检测形成鲜明的对比，边框目标检测中每个对象由粗糙的矩形框表示。在许多应用中，每个对象都需要一个二进制mask，从自动驾驶和机器人应用到照片编辑/分析，实例分割仍是一个重要的研究课题。

​		目前，实例分割的主要方法是基于检测和分割，其中，使用bounding-box检测方法对物体进行检测，然后为每个对象生成一个二进制mask。尽管过去进行了许多尝试，MaskRCNN框架是第一个在许多benchmarks中取得出色结果的框架，至今仍是使用最多的实例分割方法。尽管Mask RCNN这种方法在精度上取得了很好的结果，但是它生成的底分辨率的masks不总是可取的（如照片编辑）且以较低的帧率运行【实时性不好】，这使得注入自动驾驶一类的实时应用无法实现。

​		另一种流行的实例分割分支方法是proposal-free方法，这种方法是<span style="color:green">基于embeding损失函数或像素关联性学习。</span>由于这些方法通常依赖于密集预测网络（dense-prediction networks）。他们可以生成高分辨率的mask。除此之外，proposal-free方法通常意味着比proposal-based更快。虽然这些方法很有希望（promising），然而，他们的性能且不如Mask R-CNN。

​		在本文中，我们为proposal-free实例分割设计了一种新的损失函数，结合了两者的优点：精准，mask分辨率高结合实时性能（combined with real-time performance）。<span style="color:green">我们的方法是基于该种原则：像素可以被关联，通过指向一个物体的中心【像素与对象的关联可通过指向该对象中心来实现】。我们并未像之前的工作一样，对所有像素使用标准的回归损失（regression loss），迫使他们直接指向对象的中心不同，我们引入了新的损失函数，该函数优化了每个对象mask的iou值。因此我们的损失函数将间接的迫使对象的像素指向对象的中心。</span><span style="color:red">对于大对象，网络将学会让这个区域变大，减少远离物体中心像素的损失。（For big objects, the network will
learn to make this region bigger, relaxing the loss on pix-
els which are further away from the object’s center.）</span> <span style="color:red">在推理时，通过在每个对象中心进行聚类学习，特定于对象的区域进行聚类来恢复的。（At inference time, instances are recovered by clustering around each object’s center with the learned, object-specific region）</span>see figure 1

![image-20210311160610722](D:\Code\note\CV-Study\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210311160610722.png)

Figure 1. <span style="color:red">我们的损失函数鼓励像素指向对象中心区域的一个最佳位置，对象中心周围的特定区域，最大化每个实例对象mask的iou（Our loss function encourages pixels to point into an optimal, object-specifific region around the object’s center, maximizing the intersection-over-union of each object’s mask.）</span>对于大对象，这个区域会更大，以减少这些边缘像素的损失！左下角显示了用颜色编码学习到的偏移向量。右下角显示了移位的像素和学习道德偏移向量。<span style="color:green">通过在每个中心周围使用学习到的最佳聚类区域进行聚类来恢复实例。（ Instances are recovered by clustering around each center with the learned, optimal clustering region.）</span>

​		我们在富有挑战性的cityscapes数据集上测试了我们的方法，并且我们的方法取得了最佳效果，以27.6与26.2的得分超过了Mask R-CNN，且平均速度达到了每秒10fps。我们还观察到，我买的方法在车辆和行人上的效果很好。在cityscapes和coco上达到了与Mask R-CNN相似的分数。在cityscapes数据集上我们是第一个实时运行且精度高的方法。

​		总结：（1）<span style="color:green">提出了一种新的损失函数，这种损失函数直接优化了每个实例的iou，通过把像素拉到一个最优的，对象的特定聚类区域。（ propose a new loss function which directly optimizes the intersection-over-union of each instance by pulling pixels into an optimal, object-specifific clustering region）</span>（2）在cityscapes数据集上取得了最好的实时效果。



## Related Work

​		当前最好的实例分割方法是基于候选的（proposal-based），它依赖于Faster R-CNN对象检测框架，<span style="color:red">Faster R-CNN在当前目标检测领域处于领导地位，是benchmarks。</span>先前的实例分割方法依赖于先获得对象候选输出，在转化为实例mask。Mask R-CNN和它的衍生网络PANet 通过为Faster R-CNN网络增加了一个用于对象mask预测的分支改善和简化了this pipeline。虽然它们是流行基准（benchmarks）上最好的评分方法，但是他们生成的实例mask像素都，且并不经常用于实时应用。实例分割方法的另一个分支依赖于密集预测（dense-prediction），以输入分辨率生成实例mask的分割网络。【输入200x200，生成的mask也是200x200？】这些方法大多基于embedding损失函数，它们强迫属于同一实例像素的特征向量相互相似并且与属于其他对象的像素的特征向量完全不同【属于同一实例的像素的特征向量要相似，不同实例的要完全不同。】近期的工作表明，<span style="color:red">全卷积神经网络的空间不变性</span>对embedding方法并不理想，并建议合并坐标映射或使用所谓的半卷积来缓解这一问题。然而，这些方法仍未能达到与proposal-based一样的性能。

​		有鉴于此，Kendall等人提出了一个更有前景和简单的方法。他们提出通过指向对象的中心来为对象分配像素。这样，他们通过学习位置的相对偏移向量来避免上述的空间不变性问题。







