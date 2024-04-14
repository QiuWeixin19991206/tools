'''Unet'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import glob
from PIL import Image
print(torch.__version__)

BATCH_SIZE = 8

plt.figure()
pil_img = Image.open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset\training\00001.png')
np_img = np.array(pil_img)
plt.imshow(np_img)

plt.figure()
pil_img = Image.open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset\training\00001_matte.png')
np_img = np.array(pil_img)
plt.imshow(np_img)

print(np.unique(np_img))#0~255

#语义分割人和背景
plt.figure()
np_img[np_img>0] = 1
plt.imshow(np_img)
print(np.unique(np_img))#0~255
print(np_img.max(), np_img.min(), np_img.shape)#1 0 (800, 600)

#读取matte.png 和 png
all_pics = glob.glob(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset\training\*.png')
images = [p for p in all_pics if 'matte' not in p]#取出png
annotations = [p for p in all_pics if 'matte' in p]#取出matte.png
print(len(images), len(annotations))

#乱序训练集
np.random.seed(2021)
index = np.random.permutation(len(images))#序号
images = np.array(images)[index]
anno = np.array(annotations)[index]

#读取测试集
all_test_pics = glob.glob(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset\testing\*.png')
test_images = [p for p in all_test_pics if 'matte' not in p]
test_anno = [p for p in all_test_pics if 'matte' in p]

# 图像预处理包
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class Portrait_dataset(data.Dataset):
    def __init__(self, img_paths, anno_paths):
        self.imgs = img_paths
        self.annos = anno_paths

    def __getitem__(self, index):
        img = self.imgs[index]#切片路径
        anno = self.annos[index]

        pil_img = Image.open(img)#读取路径
        img_tensor = transform(pil_img)

        pil_anno = Image.open(anno)
        anno_tensor = transform(pil_anno)
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)#[256， 256， 1]channel=1,torch.long转整型
        anno_tensor[anno_tensor > 0] = 1#0~255, 人为二分类

        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs)
#预处理
train_dataset = Portrait_dataset(images, anno)#相当于torchvision.datasets.ImageFolder
test_dataset = Portrait_dataset(test_images, test_anno)

# 数据加载器
train_dl = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dl = data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
)

imgs_batch, annos_batch = next(iter(train_dl))
print(imgs_batch.shape, annos_batch.shape)#torch.Size([8, 3, 256, 256]) torch.Size([8, 256, 256])
img = imgs_batch[0].permute(1, 2, 0).numpy()
anno = annos_batch[0].numpy()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(anno)
# Unet
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1),#-(k-1)+1*2
            nn.ReLU(inplace=True),#中间数据被覆盖，不建议使用，但运行更快
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()

        self.conv_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels,kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        #反卷积层 padding这里是从里面第几个开始反卷积，与卷积padding不一样
        #output_padding=1，+2 因为kernel_size=3， -(k-1), 则h+2-2 w+2-2
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(channels,channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.up(x5)

        #合并，沿着dim=1的维度合并--channel维度 0 1 2 3 b, c, h, w
        x5 = torch.cat([x4, x5], dim=1)  # 32*32*1024
        x5 = self.up1(x5)  # 64*64*256)
        x5 = torch.cat([x3, x5], dim=1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = torch.cat([x2, x5], dim=1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = torch.cat([x1, x5], dim=1)  # 256*256*128

        x5 = self.conv_2(x5, is_pool=False)  # 256*256*64

        x5 = self.last(x5)  # 256*256*3
        return x5


model = Net()

if torch.cuda.is_available():#调用gpu跑模型
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()

from torch.optim import lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / (total * 256 * 256)

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
    epoch_test_acc = test_correct / (test_total * 256 * 256)

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = 1
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

# 保存模型
PATH = './model/Unet.pth'
torch.save(model.state_dict(), PATH)

# 测试模型
my_model = Net()
my_model.load_state_dict(torch.load(PATH))
num = 3#画3组图，  num*3，原图 分割图 预测图
image, mask = next(iter(test_dl))
pred_mask = my_model(image)

plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(num, 3, i * num + 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.subplot(num, 3, i * num + 2)
    plt.imshow(mask[i].cpu().numpy())
    plt.subplot(num, 3, i * num + 3)
    plt.imshow(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())#.detach()实际结果

# # 在train数据上测试
# image, mask = next(iter(train_dl))
# pred_mask = my_model(image)
#
# plt.figure(figsize=(10, 10))
# for i in range(num):
#     plt.subplot(num, 3, i * num + 1)
#     plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
#     plt.subplot(num, 3, i * num + 2)
#     plt.imshow(mask[i].cpu().numpy())
#     plt.subplot(num, 3, i * num + 3)
#     plt.imshow(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())

'''预测应用'''
num = 1 #使用几张图预测
path = r'F:\qwx\学习计算机视觉\机器学习\黑马\pantyhose.jpg'
# path = r'F:\qwx\学习计算机视觉\机器学习\黑马\mobanpipei.jpg'
pil_img = Image.open(path)#[h, w, c]
img_tensor = transform(pil_img)#预处理 此时shape=[c, h, w]
img_tensor_batch = torch.unsqueeze(img_tensor, 0)#增加一个批次维度[b, c, h, w]
pred = my_model(img_tensor_batch)

for i in range(num):
    plt.subplot(num, 3, i * num + 1)
    plt.imshow(img_tensor[i].permute(1, 2, 0).cpu().numpy())
    plt.subplot(num, 3, i * num + 2)
    plt.imshow(torch.argmax(pred[i].permute(1, 2, 0), axis=-1).detach().numpy())#.detach()实际结果

plt.show()










