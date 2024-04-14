from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel, self).__init__()

        self.layer1 = nn.Linear(20, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, input):
        x = self.layer1(input)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    '''数据处理'''
    data = pd.read_csv(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\参考代码和部分数据集\参考代码\1-18节参考代码和数据集\基础部分参考代码和数据集\daatset\HR.csv')
    data = data.join(pd.get_dummies(data.salary))
    print(pd.get_dummies(data.salary))
    del data['salary']
    data = data.join(pd.get_dummies(data.part))
    del data['part']
    y = data.left.values.reshape(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    x = [i for i in data.columns if i != 'left']  # 除left标签外的其他名称
    x = data[x].values  # 取为数据
    x = torch.from_numpy(x).type(torch.FloatTensor)

    '''超参数'''
    batch = 64
    no_of_batches = len(data)//batch
    epochs = 100
    model = Mymodel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    HR_ds = TensorDataset(x, y)

    # for epoch in range(epochs):
    #     for i in range(no_of_batches):
    #         # start = i*batch
    #         # end = start + batch
    #         # x_train = x[start: end]
    #         # y_train = y[start: end]
    #         x_train, y_train = HR_ds[i*batch,:i*batch+batch]
    #
    #         y_pred = model(x_train)
    #         loss = loss_fn(y_pred, y_train)
    #         opt.zero_grad()
    #         loss.backward()#计算损失相对于模型参数的梯度
    #         opt.step()#更新模型的参数，通过优化器更新模型的参数，使损失最小化
    #
    #     with torch.no_grad():
    #         print('epoch:', epoch, 'loss:', loss_fn(model(x), y).data.numpy())

    HR_dl = DataLoader(HR_ds, batch_size=batch, shuffle=True)#代替批次的切片操作
    for epoch in range(epochs):
        for x_train, y_train in HR_dl:
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            opt.zero_grad()
            loss.backward()#计算损失相对于模型参数的梯度
            opt.step()#更新模型的参数，通过优化器更新模型的参数，使损失最小化

        with torch.no_grad():
            print('epoch:', epoch, 'loss:', loss_fn(model(x), y).data.numpy())







