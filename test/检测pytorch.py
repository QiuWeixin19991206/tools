'''
@Project ：学习计算机视觉 
@File    ：1.py
@IDE     ：PyCharm 
@Author  ：QiuWeiXin
@Date    ：2024/3/30 19:38 
'''
import torch

print(torch.__version__)
print(torch.version.cuda) # 编译当前版本的torch使用的cuda版本号
# 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用
print(torch.cuda.is_available())



