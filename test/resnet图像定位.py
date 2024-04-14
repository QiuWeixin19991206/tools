import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
from lxml import etree
from matplotlib.patches import Rectangle
import glob
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 16
pil_img = Image.open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\Oxford-IIIT Pets Dataset\dataset/images/Abyssinian_1.jpg')
np_img = np.array(pil_img)
print(np_img.shape)

plt.imshow(np_img)
 

xml = open(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\Oxford-IIIT Pets Dataset\dataset/annotations/xmls/Abyssinian_1.xml').read()
sel = etree.HTML(xml)
#读取位置信息
width = sel.xpath('//size/width/text()')[0]
height = sel.xpath('//size/height/text()')[0]
xmin = sel.xpath('//bndbox/xmin/text()')[0]
ymin = sel.xpath('//bndbox/ymin/text()')[0]
xmax = sel.xpath('//bndbox/xmax/text()')[0]
ymax = sel.xpath('//bndbox/ymax/text()')[0]

width = int(width)
height = int(height)
xmin = int(xmin)
ymin = int(ymin)
xmax = int(xmax)
ymax = int(ymax)

plt.imshow(np_img)
rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
ax = plt.gca()
ax.axes.add_patch(rect)
 

img = pil_img.resize((224, 224))
xmin = xmin*224/width
ymin = ymin*224/height
xmax = xmax*224/width
ymax = ymax*224/height

plt.imshow(img)
rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
ax = plt.gca()
ax.axes.add_patch(rect)
 

# 创建输入
# #获取图片路径
images = glob.glob('F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\Oxford-IIIT Pets Dataset\dataset/images/*.jpg')
#获取标签路径
xmls = glob.glob('F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集\Oxford-IIIT Pets Dataset\dataset/annotations/xmls/*.xml')
xmls_names = [x.split('\\')[-1].split('.xml')[0] for x in xmls]#split('\\')[-1]取\\后最后一个文件名不带格式'Abyssinian_1'
imgs = [img for img in images
        if img.split('\\')[-1].split('.jpg')[0] in xmls_names]#遍历将被xmls_names标记过的存入imgs
print(xmls_names[:5], imgs[:5])#检测是否一一对应
scal = 224
#解析xml文件存的所有信息
def to_labels(path):
    xml = open(r'{}'.format(path)).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    height = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//bndbox/xmin/text()')[0])
    ymin = int(sel.xpath('//bndbox/ymin/text()')[0])
    xmax = int(sel.xpath('//bndbox/xmax/text()')[0])
    ymax = int(sel.xpath('//bndbox/ymax/text()')[0])
    return [xmin/width, ymin/height, xmax/width, ymax/height]
#读取到图片的位置四个角的信息
labels = [to_labels(path) for path in xmls]
out1_label, out2_label, out3_label, out4_label = list(zip(*labels))
len(out1_label), len(out2_label), len(out3_label), len(out4_label)
index = np.random.permutation(len(imgs))
images = np.array(imgs)[index]#图片 x
out1_label = np.array(out1_label).astype(np.float32).reshape(-1, 1)[index]#label的四个角 y
out2_label = np.array(out2_label).astype(np.float32).reshape(-1, 1)[index]
out3_label = np.array(out3_label).astype(np.float32).reshape(-1, 1)[index]
out4_label = np.array(out4_label).astype(np.float32).reshape(-1, 1)[index]

'''划分数据集 测试集'''
i = int(len(imgs)*0.8)
train_images = images[:i]
out1_train_label = out1_label[:i]
out2_train_label = out2_label[:i]
out3_train_label = out3_label[:i]
out4_train_label = out4_label[:i]

test_images = images[i: ]
out1_test_label = out1_label[i: ]
out2_test_label = out2_label[i: ]
out3_test_label = out3_label[i: ]
out4_test_label = out4_label[i: ]

transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
])
#定义Dataset方法 方便打包
class Oxford_dataset(data.Dataset):
    def __init__(self, img_paths, out1_label, out2_label,out3_label, out4_label, transform):
        self.imgs = img_paths
        self.out1_label = out1_label
        self.out2_label = out2_label
        self.out3_label = out3_label
        self.out4_label = out4_label
        self.transforms = transform

    def __getitem__(self, index):#切片
        img = self.imgs[index]
        out1_label = self.out1_label[index]
        out2_label = self.out2_label[index]
        out3_label = self.out3_label[index]
        out4_label = self.out4_label[index]
        pil_img = Image.open(img)
        imgs_data = np.asarray(pil_img, dtype=np.uint8)
        if len(imgs_data.shape) == 2:
            imgs_data = np.repeat(imgs_data[:, :, np.newaxis], 3, axis=2)
            img_tensor = self.transforms(Image.fromarray(imgs_data))
        else:
            img_tensor = self.transforms(pil_img)
        return (img_tensor,
                out1_label,
                out2_label,
                out3_label,
                out4_label)

    def __len__(self):
        return len(self.imgs)

train_dataset = Oxford_dataset(train_images, out1_train_label,
                               out2_train_label, out3_train_label,
                               out4_train_label, transform)
test_dataset = Oxford_dataset(test_images, out1_test_label,
                               out2_test_label, out3_test_label,
                               out4_test_label, transform)

train_dl = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True,)
test_dl = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,)

(imgs_batch, out1_batch, out2_batch, out3_batch, out4_batch) = next(iter(train_dl))

plt.figure(figsize=(12, 8))
for i,(img, label1, label2,label3,label4,) in enumerate(zip(imgs_batch[:2],
                                             out1_batch[:2],
                                             out2_batch[:2],
                                             out3_batch[:2],
                                             out4_batch[:2])):
    img = (img.permute(1,2,0).numpy() + 1)/2#还原图像，因为channel位置交换还原，再从-1~1还原到0~1
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    #注意返回顺序
    xmin, ymin, xmax, ymax = label1*224, label2*224, label3*224, label4*224,#因label=xmin/w,此时等比例还原
    rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)

# 创建定位模型
resnet = torchvision.models.resnet101(pretrained=True)#参数可改
in_f = resnet.fc.in_features
print(in_f)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_base = nn.Sequential(*list(resnet.children())[:-1])#*list(resnet.children())[:-1]解包，返回除最后一层的其他层
        self.fc1 = nn.Linear(in_f, 1)
        self.fc2 = nn.Linear(in_f, 1)
        self.fc3 = nn.Linear(in_f, 1)
        self.fc4 = nn.Linear(in_f, 1)

    def forward(self, x):
        x = self.conv_base(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4

model = Net()

if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.MSELoss()

from torch.optim import lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def fit(epoch, model, trainloader, testloader):
    total = 0
    running_loss = 0

    model.train()
    for x, y1, y2, y3, y4 in trainloader:
        if torch.cuda.is_available():
            x, y1, y2, y3, y4 = (x.to('cuda'),
                                 y1.to('cuda'), y2.to('cuda'),
                                 y3.to('cuda'), y4.to('cuda'))
        y_pred1, y_pred2, y_pred3, y_pred4 = model(x)

        loss1 = loss_fn(y_pred1, y1)
        loss2 = loss_fn(y_pred2, y2)
        loss3 = loss_fn(y_pred3, y3)
        loss4 = loss_fn(y_pred4, y4)
        loss = loss1 + loss2 + loss3 + loss4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)


    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y1, y2, y3, y4 in testloader:
            if torch.cuda.is_available():
                x, y1, y2, y3, y4 = (x.to('cuda'),
                                     y1.to('cuda'), y2.to('cuda'),
                                     y3.to('cuda'), y4.to('cuda'))
            y_pred1, y_pred2, y_pred3, y_pred4 = model(x)
            loss1 = loss_fn(y_pred1, y1)
            loss2 = loss_fn(y_pred2, y2)
            loss3 = loss_fn(y_pred3, y3)
            loss4 = loss_fn(y_pred4, y4)
            loss = loss1 + loss2 + loss3 + loss4
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)


    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'test_loss： ', round(epoch_test_loss, 3),
             )

    return epoch_loss, epoch_test_loss

epochs = 10

train_loss = []
test_loss = []

for epoch in range(epochs):
    epoch_loss, epoch_test_loss = fit(epoch, model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)

plt.figure()
plt.plot(range(1, len(train_loss)+1), train_loss, 'r', label='Training loss')
plt.plot(range(1, len(train_loss)+1), test_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
 

# 模型保存
PATH = 'location_model.pth'
torch.save(model.state_dict(), PATH)
plt.figure(figsize=(8, 24))
imgs, _, _, _, _ = next(iter(test_dl))
imgs = imgs.to('cuda')
out1, out2, out3, out4 = model(imgs)
for i in range(6):
    plt.subplot(6, 1, i+1)
    plt.imshow(imgs[i].permute(1,2,0).cpu().numpy())
    xmin, ymin, xmax, ymax = (out1[i].item()*224,
                              out2[i].item()*224,
                              out3[i].item()*224,
                              out4[i].item()*224)
    rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)

plt.show()