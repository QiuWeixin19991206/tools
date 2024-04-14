import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn

class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel, self).__init__()

        self.layer1 = nn.Linear(28*28, 120)#因为输入是( , 28*28)
        self.layer2 = nn.Linear(120, 84)
        self.layer3 = nn.Linear(84, 10)#因为label 0 1 2 所以为3

    def forward(self, input):
        x = input.view(-1, 28*28)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x



if __name__ == '__main__':

    '''数据处理'''

    transformation = transforms.Compose([
        transforms.ToTensor()#转为tensor 转换到0-1之间 把channel放在第一维度
        #transforms.Normalize()#标准化
    ])
    # 是否为训练数据 是否做变换 是否下载
    train_ds = datasets.MNIST\
        ('data/', train=True, transform=transformation, download=True)
    test_ds = datasets.MNIST \
        ('data/', train=False, transform=transformation, download=True)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)

    image, label = next(iter(train_dl))
    #[64, 1, 28, 28] [b, c, h, w]与tensorflow不同

    #查看第一张图片
    img = image[0].numpy()
    img = np.squeeze(img)
    plt.imshow(img)

    def imshow(img):
        nping = img.numpy()
        nping = np.squeeze(nping)
        plt.imshow(nping)

    plt.figure(figsize=(15, 3))
    for i, img in enumerate(image[:10]):
        plt.subplot(1, 10, i+1)
        imshow(img)



    model = Mymodel()
    loss_fn = torch.nn.CrossEntropyLoss()  # 损失函数


    def fit(epoch, model, trainloader, testloader):
        correct = 0
        total = 0
        running_loss = 0
        for x, y in trainloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                correct += (y_pred == y).sum().item()
                total += y.size(0)
                running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / total

        test_correct = 0
        test_total = 0
        test_running_loss = 0

        with torch.no_grad():
            for x, y in testloader:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)
                test_running_loss += loss.item()

        epoch_test_loss = test_running_loss / len(testloader.dataset)
        epoch_test_acc = test_correct / test_total

        print('epoch: ', epoch,
              'loss： ', round(epoch_loss, 3),
              'accuracy:', round(epoch_acc, 3),
              'test_loss： ', round(epoch_test_loss, 3),
              'test_accuracy:', round(epoch_test_acc, 3)
              )

        return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc



    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # %%
    epochs = 20
    # %%
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                     model,
                                                                     train_dl,
                                                                     test_dl)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

    plt.show()
