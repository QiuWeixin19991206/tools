
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import glob
from PIL import Image

print(torch.__version__)
print(torch.cuda.is_available())

BATCH_SIZE = 128

pil_img = Image.open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset/training/00001.png')
np_img = np.array(pil_img)
plt.imshow(np_img)


pil_img = Image.open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset/training/00001_matte.png')
np_img = np.array(pil_img)
plt.imshow(np_img)
plt.show()

print(np.unique(np_img))
print(np_img.max(), np_img.min())

print(np_img.shape)

all_pics = glob.glob(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset/training/*.png')



print(all_pics[:5])

images = [p for p in all_pics if 'matte' not in p]

print(len(images))

annotations = [p for p in all_pics if 'matte' in p]

print(len(annotations))

print(images[:5])

print(annotations[:5])

print(annotations[-5:])

np.random.seed(2021)
index = np.random.permutation(len(images))

images = np.array(images)[index]

print(images[:5])

anno = np.array(annotations)[index]

print(anno[:5])

all_test_pics = glob.glob(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\HKdataset\HKdataset/testing/*.png')

test_images = [p for p in all_test_pics if 'matte' not in p]
test_anno = [p for p in all_test_pics if 'matte' in p]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class Portrait_dataset(data.Dataset):
    def __init__(self, img_paths, anno_paths):
        self.imgs = img_paths
        self.annos = anno_paths

    def __getitem__(self, index):
        img = self.imgs[index]
        anno = self.annos[index]

        pil_img = Image.open(img)
        img_tensor = transform(pil_img)

        pil_anno = Image.open(anno)
        anno_tensor = transform(pil_anno)
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)
        anno_tensor[anno_tensor > 0] = 1

        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs)



train_dataset = Portrait_dataset(images, anno)

test_dataset = Portrait_dataset(test_images, test_anno)

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

img = imgs_batch[0].permute(1, 2, 0).numpy()
anno = annos_batch[0].numpy()

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(anno)



# 创建 LinkNet 模型

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,k_size=3,stride=1,pad=1):
        super(ConvBlock, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=k_size,
                      stride=stride,
                      padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, pad=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,#stride=2放大2倍
                                         kernel_size=k_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_act=True):
        x = self.deconv(x)
        if is_act:
            x = torch.relu(self.bn(x))
        return x


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock, self).__init__()
        self.conv1_1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv1_2 = ConvBlock(out_channels, out_channels)
        self.conv2_1 = ConvBlock(out_channels, out_channels)
        self.conv2_2 = ConvBlock(out_channels, out_channels)
        self.shortcut = ConvBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv2_1(out1)
        residue = self.shortcut(x)
        out2 = self.conv2_1(out1 + residue)
        out2 = self.conv2_2(out2)
        return out2 + out1

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4,
                               k_size=1, pad=0)
        self.deconv = DeconvBlock(in_channels // 4, in_channels // 4)
        self.conv2 = ConvBlock(in_channels // 4, out_channels,
                               k_size=1, pad=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.init_conv = ConvBlock(3, 64,
                                   k_size=7,
                                   stride=2,
                                   pad=3)
        self.init_maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.encode1 = EncodeBlock(64, 64)
        self.encode2 = EncodeBlock(64, 128)
        self.encode3 = EncodeBlock(128, 256)
        self.encode4 = EncodeBlock(256, 512)

        self.decode4 = DecodeBlock(512, 256)
        self.decode3 = DecodeBlock(256, 128)
        self.decode2 = DecodeBlock(128, 64)
        self.decode1 = DecodeBlock(64, 64)

        self.deconv_last1 = DeconvBlock(64, 32)
        self.conv_last = ConvBlock(32, 32)
        self.deconv_last2 = DeconvBlock(32, 2,
                                        k_size=2,
                                        pad=0,
                                        padding=0)

    def forward(self, x):
        x = self.init_conv(x)  # (6, 128, 128, 64)
        x = self.init_maxpool(x)  # (6, 64, 64, 64)

        e1 = self.encode1(x)  # (6, 32, 32, 64)
        e2 = self.encode2(e1)  # (6, 16, 16, 128)
        e3 = self.encode3(e2)  # (6, 8, 8, 256)
        e4 = self.encode4(e3)  # (6, 4, 4, 512)

        d4 = self.decode4(e4) + e3
        d3 = self.decode3(d4) + e2
        d2 = self.decode2(d3) + e1
        d1 = self.decode1(d2)

        f1 = self.deconv_last1(d1)
        f2 = self.conv_last(f1)
        f3 = self.deconv_last2(f2, is_act=False)

        return f3


model = Net()

loss_fn = nn.CrossEntropyLoss()

# IOU 指标
from torch.optim import lr_scheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []

    model.train()
    for x, y in trainloader:
        #        if torch.cuda.is_available():
        #            x, y = x.to('cuda'), y.to('cuda')
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

            intersection = torch.logical_and(y, y_pred)#iou交集
            union = torch.logical_or(y, y_pred)#并集
            batch_iou = torch.true_divide(torch.sum(intersection),
                                          torch.sum(union))
            epoch_iou.append(batch_iou)

    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / (total * 256 * 256)

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            #            if torch.cuda.is_available():
            #                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = torch.true_divide(torch.sum(intersection),
                                          torch.sum(union))
            epoch_test_iou.append(batch_iou)

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total * 256 * 256)

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'IOU:', round(np.mean(epoch_iou), 3))
    print()
    print('     ', 'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
          'test_iou:', round(np.mean(epoch_test_iou), 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


epochs = 40
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
PATH = './model/linknet_model.pth'
torch.save(model.state_dict(), PATH)

# 测试模型
my_model = Net()
my_model.load_state_dict(torch.load(PATH))
num = 3
image, mask = next(iter(test_dl))
pred_mask = my_model(image)

plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(num, 3, i * num + 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.subplot(num, 3, i * num + 2)
    plt.imshow(mask[i].cpu().numpy())
    plt.subplot(num, 3, i * num + 3)
    plt.imshow(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())
# 在train数据上测试
image, mask = next(iter(train_dl))
pred_mask = my_model(image)

plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(num, 3, i * num + 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.subplot(num, 3, i * num + 2)
    plt.imshow(mask[i].cpu().numpy())
    plt.subplot(num, 3, i * num + 3)
    plt.imshow(torch.argmax(pred_mask[i].permute(1, 2, 0), axis=-1).detach().numpy())
plt.show()


























