# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os

torchvision.datasets.ImageFolder  # 从分类的文件夹中创建dataset数据
base_dir = r'./datasets2/4weather'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
# 图像预处理包
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
# %%
train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=transform
)
# %%
test_ds = torchvision.datasets.ImageFolder(
    test_dir,
    transform=transform
)
# %%
BTACH_SIZE = 32
# 数据加载器
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BTACH_SIZE,
    shuffle=True
)
# %%
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BTACH_SIZE,
)
# %%
model = torchvision.models.resnet18(pretrained=True)

# print(model)

for param in model.parameters():
    param.requires_grad = False

in_f = model.fc.in_features
model.fc = nn.Linear(in_f, 4)

if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    #exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
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

epochs = 50

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



plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
plt.legend()
# %%
plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
plt.legend()

plt.show()












