import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

data = pd.read_csv(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\参考代码和部分数据集\参考代码\1-18节参考代码和数据集\基础部分参考代码和数据集\daatset\HR.csv')
print(data.info)
print(data.part.unique())
print(data.groupby(['salary', 'part']).size())
#把salary下的情况分类为3中情况[high  low  medium] 方便分类
data = data.join(pd.get_dummies(data.salary))
print(pd.get_dummies(data.salary))
del data['salary']
data = data.join(pd.get_dummies(data.part))
del data['part']
print(data.head())

y = data.left.values.reshape(-1, 1)
y = torch.from_numpy(y).type(torch.FloatTensor)
print(y.shape)
x = [i for i in data.columns if i != 'left']#除left标签外的其他名称
x = data[x].values#取为数据
x = torch.from_numpy(x).type(torch.FloatTensor)
print(x.shape)



