import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import shutil#拷贝图片
from torchvision import transforms
# 此处使用训练代码
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
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


torchvision.datasets.ImageFolder  # 从分类的文件夹中创建dataset数据
#创建路径
base_dir = r'./datasets2/4weather'
#判断对象是否为目录
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')#创建目录
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
# 创建目录
specises = ['cloudy', 'rain', 'shine', 'sunrise']

for train_or_test in ['train', 'test']:
    for spec in specises:
        os.mkdir(os.path.join(base_dir, train_or_test, spec))

image_dir = r'./dataset'#图片目录
#拷贝图片
for i, img in enumerate(os.listdir(image_dir)):
    for spec in specises:#分类
        if spec in img:#判断四种天气是否在img的名字中
            s = os.path.join(image_dir, img)#拼接目录/dataset/cloudy(可变的)
            if i % 5 == 0:#6个里存1个在/dataset/test/cloudy(可变的)/
                d = os.path.join(base_dir, 'test', spec, img)
            else:
                d = os.path.join(base_dir, 'train', spec, img)
            shutil.copy(s, d)#复制d文件在s路径下
#打印各个目录下图片数量
for train_or_test in ['train', 'test']:
    for spec in specises:
        print(train_or_test, spec, len(os.listdir(os.path.join(base_dir, train_or_test, spec))))

#图片操作
transform = transforms.Compose([
    transforms.Resize((96, 96)),#图片转换统一大小
    transforms.ToTensor(),#channel放前面
    transforms.Normalize(mean=[0.5, 0.5, 0.5],#三个维度 猜测均值0.5 方差0.5
                         std=[0.5, 0.5, 0.5])
])
#加载数据集
train_ds = torchvision.datasets.ImageFolder(train_dir,transform=transform)
test_ds = torchvision.datasets.ImageFolder(test_dir,transform=transform)

#train_ds.classes： ['cloudy', 'rain', 'shine', 'sunrise']
#train_ds.class_to_idx：{'cloudy' : o, 'rain': 1, 'shine’: 2， 'sunrise’: 3
#len(train_ds), len(test_ds) ：900， 225

BATCHSIZE = 16

train_dl = torch.utils.data.DataLoader(train_ds,batch_size=BATCHSIZE,shuffle=True)
# %%
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=BATCHSIZE)

imgs, labels = next(iter(train_dl))

#画一张图
im = imgs[0].permute(1, 2, 0)#交换channel顺序
im = im.numpy()#转np
im = (im + 1) / 2 #-1~1还原为0~1
im.max(), im.min()
plt.imshow(im)

print(labels[0])
print(train_ds.class_to_idx)

id_to_class = dict((v, k) for k, v in train_ds.class_to_idx.items())
#拼接几张图一起画
plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs[:6], labels[:6])):
    img = (img.permute(1, 2, 0).numpy() + 1) / 2
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_class.get(label.item()))
    plt.imshow(img)


# # %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop(x)
        #        print(x.size())
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = Net()
preds = model(imgs)
torch.argmax(preds, 1)
#放gpu上运行
if torch.cuda.is_available():
    model.to('cuda')


loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30

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
# %%
plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
plt.legend()
# %%
plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
plt.legend()


# %% md
# 添加dropout层
# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


# %%
model = Net()
if torch.cuda.is_available():
    model.to('cuda')
    print('gpu')
# %%
optim = torch.optim.Adam(model.parameters(), lr=0.001)
# %%
epochs = 30
# %%
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

fig3 = plt.figure()
plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
plt.legend()

plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
plt.legend()

plt.show()