import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel, self).__init__()

        self.layer1 = nn.Linear(4, 32)#因为输入是( , 4)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 3)#因为label 0 1 2 所以为3

    def forward(self, input):
        x = self.layer1(input)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

'''计算正确率'''
def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)#[0.1, 0.2, 0.7]计算除位置后，得到pred_label=3
    acc = (y_pred == y_true).float().mean()
    return acc

if __name__ == '__main__':

    '''数据处理'''
    data = pd.read_csv(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\参考代码和部分数据集\参考代码\1-18节参考代码和数据集\基础部分参考代码和数据集\daatset\iris.csv')
    print(data)
    data['Species'] = pd.factorize(data.Species)[0]#把几种分类
    print(data)#[150 rows x 6 columns]

    x = data.iloc[:, 1:-1].values#只取值，不要标题 (150, 4)
    y = data.iloc[:, -1].values#(150,)
    # y = y.reshape(-1, 1)#(150,1)
    print(x.shape, y.shape)#(150, 4) (150, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = torch.Tensor(x_train).type(torch.float32)
    x_test = torch.Tensor(x_test).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    y_test = torch.Tensor(y_test).type(torch.int64)

    '''超参数'''
    batch = 8
    epochs = 100
    model = Mymodel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False)

    input_batch, label_batch = next(iter(train_dl))#返回一个批次的张量, 代替了第二次for运算 遍历一个batch

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(epochs):
        for x, y in train_dl:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
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
            train_loss.append(loss)
            train_acc.append(acc)
            test_loss.append(loss1)
            test_acc.append(acc1)

    fig1 = plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
    plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
    plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
    plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
    plt.legend()

    plt.show()
