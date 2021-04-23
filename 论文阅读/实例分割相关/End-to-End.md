# 汇报

- 第一步
    - 介绍传统的两步走的目标检测方法，Mask R-CNN。 
        - 需要定义大量的proposals，anchors或window centers。
        - 这里的很多设计涉及到先验知识。（常识：举例 anchors大小的选择）
        - 这些proposals anchors可能会有很多重叠的部分。
    - 对比介绍`DERT`网络，于Mask R-CNN相比，DERT的优势在哪里。
        - 结构更简单，没有proposals anchors 和 window centers等。直接预测需要检测的目标对象的集合。
        - DERT直接预测最终的目标集合，通过CNN+transformer结构。
        - 在训练期间，用二分图匹配算法对某个对象进行唯一的预测。每匹配到的则归类为空对象（背景）

- 第二步，介绍dert网络的思想

    - 将目标检测问题视为直接预测一个集合。集合中的值就是需要检测的那些对象。
    - 采用流行的基于transformers encoder-decoder结构。
        - transformer的self-attention机制明确的对序列中元素之间的所有成对交互（pairwise interactions 成对关系）进行建模，使得这些架构特别适合有特定约束的集合预测（）。
    - DERT一次预测所有物体，并且用一个集合损失函数进行端到端的训练，该函数在预测物体和ground truth之间进行二分匹配
    - 简化了检测的过程，丢弃了spatial anchors or non-maximal suppression。

    