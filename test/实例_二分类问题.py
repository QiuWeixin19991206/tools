import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel, self).__init__()

        self.layer1 = nn.Linear(20, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, input):
        x = self.layer1(input)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x

'''计算正确率'''
def accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).type(torch.int32)
    acc = (y_pred == y_true).float().mean()
    return acc

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

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    '''超参数'''
    batch = 64
    epochs = 100
    model = Mymodel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    '''包装'''
    x_train = torch.Tensor(x_train).type(torch.float32)
    x_test = torch.Tensor(x_test).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False)

    for epoch in range(epochs):
        for x_train, y_train in train_dl:
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            opt.zero_grad()
            loss.backward()#计算损失相对于模型参数的梯度
            opt.step()#更新模型的参数，通过优化器更新模型的参数，使损失最小化

        with torch.no_grad():
            acc = accuracy(model(x_train), y_train)
            loss = loss_fn(model(x_train), y_train).data.numpy()

            acc1 = accuracy(model(x_test), y_test)
            loss1 = loss_fn(model(x_test), y_test).data.numpy()
            print('epoch:', epoch, '   train:   loss:', np.round(loss, 3), 'accuracy:', np.round(acc.numpy(), 3),
                  '   test:   loss:', np.round(loss1, 3), 'accuracy:', np.round(acc1.numpy(), 3))#np.round( , 3)保留3位小数








