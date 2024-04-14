
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split

'''数据文本处理'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv(r'F:\qwx\学习计算机视觉\行\pytorch全套入门与实战项目\课程资料\文本分类数据集\文本分类数据集\Tweets.csv')
print(data.head())
data = data[['airline_sentiment', 'text']]#只要这两列
print(data.head())
data.airline_sentiment.unique()# 获取'airline_sentiment'列的唯一值
print(data.head())
print(data.airline_sentiment.value_counts())## 统计每种情感类别的出现次数
print(data.head())
data['review'] = pd.factorize(data.airline_sentiment)[0]# 将'airline_sentiment'列编码为数字值
print(data.head())
del data['airline_sentiment']# 从数据集中删除'airline_sentiment'列
print(data.head())
token = re.compile('[A-Za-z]+|[!?,.()]')# 使用正则表达式的分词函数，将文本分割成单词或者标点符号
def reg_text(text):
    new_text = token.findall(text)#使用之前编译好的正则表达式对象 token，在给定的文本 text 中找到所有匹配的模式（即单词或标点符号）
    new_text = [word.lower() for word in new_text]# 将文本转换为小写并进行分词处理
    return new_text

data['text'] = data.text.apply(reg_text)#将 reg_text 函数应用到 data['text'] 列的每个元素上
print(data.head())
# 构建词汇表
# 遍历data['text'] 中的每个文本。对于每个文本，又遍历文本中的每个单词，并将每个单词添加到word_set 集合中。
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)

max_word = len(word_set) + 1 #计算词汇表中单词的数量，并将其加1，以便为填充值留出一个位置。
word_list = list(word_set) #将词汇表转换为列表，方便后续操作
word_index =  dict((word, word_list.index(word) + 1) for word in word_list)# 构建单词到索引的映射字典
text = data.text.apply(lambda x: [word_index.get(word, 0) for word in x])# 将文本转换为索引序列,如果单词不在词汇表中，则用索引0表示。
maxlen = max(len(x) for x in text) #计算文本中最长的序列长度
pad_text = [l + (maxlen-len(l))*[0] for l in text]# 对文本进行填充，使其长度相同.[14640, 40]因为最长的一句话转为索引后有40个数字
pad_text = np.array(pad_text) #将填充后的文本转换为 NumPy 数组。
labels = data.review.values #提取情感标签

x_train, x_test, y_train, y_test = train_test_split(pad_text, labels)
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)

train_ds = Mydataset(x_train, y_train)
test_ds = Mydataset(x_test, y_test)

BTACH_SIZE = 16
train_dl = torch.utils.data.DataLoader(
                                       train_ds,
                                       batch_size=BTACH_SIZE,
                                       shuffle=True
)
test_dl = torch.utils.data.DataLoader(
                                       test_ds,
                                       batch_size=BTACH_SIZE
)
# Embeding : 把文本映射为一个密集向量
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(max_word, 100)   # batch*maxlen*100
        self.fc1 = nn.Linear(maxlen*100, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.em(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to('cuda')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
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
#    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total


    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
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

epochs = 10

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
plt.figure()
a = np.arange(epochs)
plt.plot(np.arange(epochs), train_acc, c='r', label='train_acc')
plt.plot(np.arange(epochs), test_acc, c='b', label='test_loss')
plt.legend()
plt.figure()
plt.plot(np.arange(epochs), train_loss, c='r', label='train_loss')
plt.plot(np.arange(epochs), test_loss, c='b', label='test_loss')
plt.legend()

plt.show()
# # 使用LSTM
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.em = nn.Embedding(max_word, 100)   # batch*maxlen*100
#         self.lstm = nn.LSTM(100, 200, batch_first=True)
#         self.fc1 = nn.Linear(200, 256)
#         self.fc2 = nn.Linear(256, 3)
#
#     def forward(self, x):
#         x = self.em(x)
#         x, _ = self.lstm(x)
#         x = F.relu(self.fc1(x[:, -1, :]))
#         x = self.fc2(x)
#         return x
# #%%
# model = Net().to(device)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# epochs = 20
#
# train_loss = []
# train_acc = []
# test_loss = []
# test_acc = []
#
# for epoch in range(epochs):
#     epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
#                                                                  model,
#                                                                  train_dl,
#                                                                  test_dl)
#     train_loss.append(epoch_loss)
#     train_acc.append(epoch_acc)
#     test_loss.append(epoch_test_loss)
#     test_acc.append(epoch_test_acc)
# plt.figure()
# plt.plot(np.range(epochs), train_acc, c='r', label='train_acc')
# plt.plot(np.range(epochs), test_acc, c='b', label='test_loss')
# plt.legend()
#
# plt.figure()
# plt.plot(np.range(epochs), train_loss, c='r', label='train_loss')
# plt.plot(np.range(epochs), test_loss, c='b', label='test_loss')
# plt.legend()
