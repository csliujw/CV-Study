# 安装记录`PyTorch

- 安装miniconda <a href="ModuleNotFoundError: No module named 'torch._C'">下载地址</a>
- 下载pytorch的离线包，选CPU版本的，离线安装。在线安装我试了好多次，总出错，最后还是去清华镜像下载的。<a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/">下载地址</a>
- 下载离线安装包 <a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/">清华镜像下载地址</a>
  - 注意conda环境中的py的版本 一定要一致，不一致会出错！！！
- conda install 下载的压缩包
- over

```python
# 测试代码
import torch

if __name__ == '__main__':
    print(123)
    print(torch.cuda.is_available())
    
# output
# 123
# True
```

