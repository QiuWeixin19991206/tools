import matplotlib.pyplot as plt
import torch
import torchvision
import glob
import numpy as np
from PTL import Image#打开图片
from torch.utils import data

# img_path = glob.glob('F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\大型数据集\大型数据集/birds/birds/*/*.jpg')
#
# img_p = img_path[100]
# print(img_p)
# img_p.split('\\')
# print(img_p)
# img_p.split('.')
# print(img_p)
#
# all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in img_path]
# print(all_labels_name)



class BirdDataset(data.Dataset):

    def __init__(self, imgs_path, labels):
        self.imgs = imgs_path
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        pil_img = Image.open(img)
        np_img = np.asarray(pil_img, dtype=np.uint8)
        if len(np_img.shape)==2:#防止灰度图读入进来影响运行
            img_data = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)#对第三维度复制
            img_data = Image.fromarray(img_data)
        img_tensor = tranform(pil_img)

        return img_tensor, label

    def __len__(self):
        return len(self.imgs)

train_ds = BirdDataset(train_path, train_labels)
train_dl = data.DataLoader(train_ds, batch_size=64, shuffle=True)
img_batch, label_batch = next(iter(train_dl))
#[32, 3, 224, 224]
plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(img_batch[:6], label_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
    plt.title(index_to_label.get(label.item()))
    plt.imshow(img)



