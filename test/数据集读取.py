import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


data = pd.read_csv(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\参考代码和部分数据集\参考代码\1-18节参考代码和数据集\基础部分参考代码和数据集\daatset\Income1.csv')
print(data.info)
# plt.scatter(data.Education, data.Income)
# plt.show()

X = data.Education.values.reshape(-1, 1).astype(np.float32)
Y = data.Income.values.reshape(-1, 1).astype(np.float32)
X = torch.from_numpy(X)#torch的np转tensor
Y = torch.from_numpy(Y)

'''包装方法'''
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
for epoch in range(1000):
    for x, y in zip(X, Y):
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        opt.zero_grad()#把权重清零
        loss.backward()#反向传播 求解梯度
        opt.step()#优化参数

fig1 = plt.figure()
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), model(X).data.numpy(), c='r')

'''源代码'''
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
learning_rate = 1e-3
for epoch in range(1000):
    for (x, y) in zip(X, Y):
        y_pred = torch.matmul(x, w) + b
        loss = (y - y_pred).pow(2).mean()
        #把梯度置为0 否则每次会继续累积
        if not w.grad is None:
            w.grad.data.zero_()#就地改为0
        if not b.grad is None:
            b.grad.data.zero_()  # 就地改为0
        loss.backward()
        with torch.no_grad():#梯度下降 无需跟踪, 梯度更新
            w.data -= w.grad.data * learning_rate
            b.data -= b.grad.data * learning_rate

fig2 = plt.figure()
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), (X * w + b).data.numpy(), c='r')

fig3 = plt.figure()
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), model(X).data.numpy(), c='r', linewidth=5)
plt.plot(X.numpy(), (X * w + b).data.numpy(), c='b')

plt.show()







