## DETR结构

整个DETR的结构十分简单，如图二所示。它包含三个主要组件。

- CNN backbone提取compact特征表示
- encoder-decoder transformer
- 一个进行最终检测的简单前馈网络（FFN）

![image-20210417160102874](D:\Code\note\CV-Study\pics\CV\ISG\DETR结构\DETR.png)

> Backbone

初始图像大小$x_{img}∈ \mathbb{R}^{C*H*W}$（三通道的彩色图片），$C=2048$ and $H,W = \frac{H_0}{32},\frac{W_0}{32}$

> transformer encoder

首先，$1*1$的卷积将high-level activation map f从C个通道减少到更小的维数d，从而生成一个新的特征图$z_0∈\mathbb{R}^{d*H*W}$,编码器需要一个序列作为输入，会将$z_0$的空间尺寸折叠为一维的，从而生成$d*HW$的特征图。每个编码层都有一个标准的结构，由一个multi-head self-attention模块和一个前馈网络（FFN）组成。因为transormer是`特征之间没有空间位置关系`，因此`提供了一个固定位置编码来进行补充`，该编码被添加到每个attention层中。

> transformer decoder

解码器遵循标准的transformer结构，使用multi-headed self- and encoder-decoder attention机制。`与原始的转换器的不同之处在于，该模型在每个解码器并行解码N个对象`。由于解码器也是`特征之间没有空间位置关系`，因此N个输入嵌入必须不同才能产生不同的结果。这些输入嵌入是学习的位置编码，我们将其称为对象查询，并且类似于编码器，我们将它们添加到每个关注层的输入中。 N个对象查询由解码器转换为嵌入的输出。 然后，它们通过前馈网络（在下一个小节中描述）独立地解码为框坐标和类标签，从而得出N个最终预测。 通过对这些嵌入的自编码器和解码器注意，模型可以使用它们之间的成对关系全局地将所有对象归为一类，同时能够将整个图像用作上下文。 

> FFNs（Prediction feed-forward networks）

最终的预测是通过一个3层感知器和一个线性投影层（linear projection layer）计算，感知器具有ReLU激活函数和d维隐藏层。FFN预测bounding box归一化的中心坐标、图像的高度和宽度，线性层使用softmax函数预测类标签。由于我们预测一个固定大小的N个包围框，其中`N>需要检测的数目`，因此使用额外的特殊类标签∅表示槽内没有检测到对象。

> Auxiliary decoding losses

我们发现在训练期间在解码器中使用辅助损耗[1]很有帮助，特别是有助于模型输出正确数量的每个类的对象。 我们在每个解码器层之后添加预测FFN和匈牙利损失。 所有预测FFN均共享其参数。 我们使用附加的共享层范数来标准化来自不同解码器层的预测FFN的输入。 

---

