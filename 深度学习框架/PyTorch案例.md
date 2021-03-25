# PyTorch案例

## CIFAR-10案例

> **定义网络**

使用LeNet网络架构

```python
"""
pytorch的通道排序
[batch, channel, height, width]
"""
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # input(3,32,32)  卷积后 output(16,28,28) debug看就行
        x = self.pool1(x)  # output(16,14,14)

        x = torch.relu(self.conv2(x))  # output(32,10,10)
        x = self.pool2(x)  # output(32,5,5)

        # view 展平。 下面的代码是展平为一维向量。-1表示自动推理。最后是 1*800的向量
        # 原本是 [batch, channel, height, width]  -1就表示为batch自动推理
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = torch.relu(self.fc1(x))  # output(120)
        x = torch.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        # softmax 一般会将输出转换为概率值，会在最后一层加softmax。但是在交叉熵函数中包含了softmax这个过程，且更为高效，故这里不用softmax了
        return x


if __name__ == '__main__':
    input1 = torch.rand([32, 3, 32, 32])
    model = LeNet()
    print(model)
    output = model(input1)
```

> **未重构带注释的代码**

```python
"""
在本教程中，我们将使用 CIFAR10 数据集。
它具有以下类别：“飞机”，“汽车”，“鸟”，“猫”，“鹿”，“狗”，“青蛙”，“马”，“船”，“卡车”。
CIFAR-10 中的图像尺寸为3x32x32，即尺寸为32x32像素的 3 通道彩色图像。

如何导入CIFAR-10数据集？用torchvision包下的datasets下的CIFAR10。[这个了解就行]
"""
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from LeNet import LeNet

transform = transforms.Compose([
    # 把图片数据转为tensor，这样torch才可以训练
    # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    # 更加具体的去看下源码注释
    transforms.ToTensor(),
    # 标准化过程，具体看源码注释
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DOWNLOAD = False
train_set = datasets.CIFAR10(root="./data", train=True, download=DOWNLOAD, transform=transform)
test_set = datasets.CIFAR10(root="./data", train=False, download=DOWNLOAD, transform=transform)

tranLoader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=0)

# 将测试集转为可迭代的对象 目的是为了看一下图片奥，没其他意思！
test_data_iter = iter(testLoader)
# 这个是它返回值的顺序，记住就好
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == '__main__':
    model = LeNet()
    # This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.
    # CrossEntropyLoss包含了softmax的过程，不用我们自己在网络的最后一层手动softmax了！！
    # 如果你想自己在网络中手动softmax，那么就用NLLLoss损失函数吧，这个没有包含softmax的过程
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        running_loss = 0.0
        """
            enumerate is useful for obtaining an indexed list:
            (0, seq[0]), (1, seq[1]), (2, seq[2]), ...
            步长，数据 in enumerate(tranLoader, 开始步长=0):
        """
        for step, data in enumerate(tranLoader, start=0):
            inputs, labels = data
            """
            为什么每计算一个batch 就需要调用一次optimizer.zero_grad()
            如果不清除历史梯度，就会对计算的历史梯度进行累计（通过这个特性可以变相实现一个很大的batch数值的训练）
            https://www.zhihu.com/question/303070254
            """
            optimizer.zero_grad()
            outputs = model(inputs)
            # 计算损失
            loss = loss_function(outputs, labels)
            # 反向传播
            loss.backward()
            # 通过优化器进行参数更新
            optimizer.step()
            running_loss += loss.item()

            if step % 500 == 499:
                # with 上下文管理器。在验证、预测的时候用这个torch.no_grad(),不会计算误差梯度了！
                with torch.no_grad():
                    outputs = model(test_image)  # [batch, 10]
                    # 我们需要在维度1上寻找最大值，因为维度0是batch，维度1才是各个类别的概率
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f   test_accuracy: %.3f ' % (
                        epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    # 保存网络参数
    print("Finished Training")
    torch.save(model.state_dict(), 'leNet.pth')

# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
# def imgshow(img):
#     img = img / 2 + 0.5  # 对图像进行反标准化处理
#     npimg = img.numpy()  # tensor转numpy
#     # transpose 通道转换 从 channel height width --> height width channel
#     # channel   原来为位置是0
#     # height    原来为位置是1
#     # width     原来为位置是1
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
#
# imgshow(torchvision.utils.make_grid(test_image))
```

> 已重构的代码+tqdm+parser

- tqdm：进度条
- parser：控制台字符解析

```python
"""
在本教程中，我们将使用 CIFAR10 数据集。
它具有以下类别：“飞机”，“汽车”，“鸟”，“猫”，“鹿”，“狗”，“青蛙”，“马”，“船”，“卡车”。
CIFAR-10 中的图像尺寸为3x32x32，即尺寸为32x32像素的 3 通道彩色图像。

如何导入CIFAR-10数据集？用torchvision包下的datasets下的CIFAR10。[这个了解就行]
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from LeNet import LeNet

"""
相关常量
"""
DOWNLOAD = False

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def transform_data():
    transform = transforms.Compose([
        # 把图片数据转为tensor，这样torch才可以训练
        # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
        # 更加具体的去看下源码注释
        transforms.ToTensor(),
        # 标准化过程，具体看源码注释
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def load_data():
    train_set = datasets.CIFAR10(root="./data", train=True, download=DOWNLOAD, transform=transform_data())
    test_set = datasets.CIFAR10(root="./data", train=False, download=DOWNLOAD, transform=transform_data())
    """
    trainLoader 有如下属性值：
        batch_size = 24
        trainLoader是可迭代的
    """
    trainLoader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=0)
    testLoader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=0)
    # print(f"train total data size is {len(trainLoader) * trainLoader.batch_size}")
    # print(f"test total data size is {len(testLoader) * testLoader.batch_size}")

    return trainLoader, testLoader


def train_one_epoch(model, loss_function, optimizer, trainLoader, epoch, running_loss=0.0):
    # ============ 进度条显示训练进度 ============
    par = tqdm(total=len(trainLoader))
    par.set_description(f'train epoch {epoch} \t')
    # ============ 进度条显示训练进度 ============

    for step, data in enumerate(trainLoader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # ============ 进度条优化输出 ============
        postfix = OrderedDict([('loss', loss.item()), ])
        par.set_postfix(postfix)
        par.update(1)
        # ============ 进度条优化输出 ============
    par.close()
    return running_loss / len(trainLoader)


@torch.no_grad()
def test(model, testLoader, epoch, accuracy=0):
    model.eval()

    # ============ 进度条显示训练进度 ============
    par = tqdm(total=len(testLoader))
    par.set_description(f'test epoch {epoch} \t')
    # ============ 进度条显示训练进度 ============

    for step, data in enumerate(testLoader, start=0):
        inputs, lables = data
        outputs = model(inputs)
        predict_y = torch.max(outputs, dim=1)[1]
        accuracy += (predict_y == lables).sum().item()
        par.update(1)

    par.close()
    total_data_size = len(testLoader) * testLoader.batch_size
    return accuracy / total_data_size


def main():
    config = assign()
    model = LeNet()
    trainLoader, testLoader = load_data()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['epoch']):
        loss = train_one_epoch(model, loss_function, optimizer, trainLoader, epoch)
        accuracy = test(model, testLoader, epoch)
        print(f"\ncurrent {epoch} epoch loss={loss} accuracy={accuracy}\n")
    # 保存网络参数
    print("Finished Training")
    torch.save(model.state_dict(), config['save_dir'] + config['save_name'])


def parse_args():
    import argparse
    # 控制台输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, help='保存训练参数的目录')
    parser.add_argument('--save_name', default=None, help='保存训练参数的名字')
    parser.add_argument('--epoch', default=1, help='默认训练轮数，默认为50')
    # 将相关参数变为字典
    return vars(parser.parse_args())


def assign():
    config = parse_args()
    if config['epoch'] is None:
        config['epoch'] = 10
    if config['save_dir'] is None:
        config['save_dir'] = './'
    if config['save_name'] is None or config['save_name'] == '':
        config['save_name'] = 'model.pth'
    return config


if __name__ == '__main__':
    main()
```

> **预测代码**

```python
import torch
from LeNet import LeNet
from torchvision.transforms import transforms

transforms.Compose([
    # 我们下载的图片大小要调整成32*32，因为网络训练的输入图片大小就是32*32
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
net.eval()
net.load_state_dict(torch.load('lenet.pth'))
```

## 大批量训练

- torch.no_grad()：是不进行求梯度。避免在验证/val/test的时候消耗过多显存，减少显存的开销。
- optimizer.zero_grad() pytorch需要进行梯度清零，不清零的话梯度会进行累加。
  - 因此，我们可以通过这个特点，变相实现大批量的训练。具体代码demo如下：

```python
for i,(images,target) in enumerate(train_loader):
    images = images.cuda()
    target = target.cuda()
    outputs = model(images)
    loss = criterion(outputs,target)
    # accumulation_steps 次 for 的总数算一个big batchsize
    # 所以我们求每次loss的平均值。 大致意思就是 (loss_1 / 5 + loss_2 / 5+ loss_3 / 5 + loss_4 / 5 + loss_5 / 5)
    loss = loss / accumulation_steps
    loss.backward()
    
    if ( (i+1) % accumulation_steps ) == 0:
        optimizer.step()
        optimizer.zero_grad()
```



