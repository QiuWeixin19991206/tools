# import torch.nn
import torchtext
import numpy as np
from torchtext.vocab import GloVe

'''例如：pip install numpy　-i　 https://mirrors.aliyun.com/pypi/simple/

　　pip install 包名==版本号

例如：pip install numpy==1.20.3

附国内常用源镜像地址
清华：https://pypi.tuna.tsinghua.edu.cn/simple

阿里云：http://mirrors.aliyun.com/pypi/simple/

豆瓣(douban) ：http://pypi.douban.com/simple/

中国科技大学 https://pypi.mirrors.ustc.edu.cn

华中理工大学：http://pypi.hustunique.com/

山东理工大学：http://pypi.sdutlinux.org/
'''



text = torchtext.data.Field(lower=True, fix_length=200, batch_first=True)
label = torchtext.data.Field(sequential=False)
train, test = torchtext.datasets.IMDB.splits(text, label)
'''创建词表'''
#只为1w个单词创建词表，其余没有,freq=3视为低于3次出现的等于没有
text.build_vocab(train, max_size=10000, min_freq=10, vectors=None)
label.build_vocab(train)
train_iter, test_iter = torchtext.data.BucketIterator.splits((train,test),batch_size=16)
b = next(iter(train_iter))#16,

embeding_dim = 100 #把每一个单词映射成长度为100的张量
hidden_size = 200 #状态值的大小
# batch_size * 200 * 100 #seq_length(序列) * batch *embeding_dim

class RNN_Ecoder(nn.Module): #这个模型将对评论一次读取，并输出最后状态
    def __init__(self, input_seq_length, hidden):
        super(RNN_Ecoder, self).__init__()
        self.rnn = nn.RNNCell(input_seq_length, hidden)

    def forward(self, inputs): #input是输入序列  seq序列长度, batch, embeding
        bz = inputs.shape[1]
        ht = torch.zeros(bz, self.hidden)
        for word in inputs:
            ht = self.rnn(word, ht)#ht上一个状态输出
        return ht

class Net(nn.Module):
    def __init__(self, vocab_size, em_dim, input_seq_length, hidden_size):
        super(Net, self).__init__()
        self.em = nn.Embedding(vocab_size, em_dim) #单词的大小
        self.rnn = RNN_Ecoder(input_seq_length, hidden_size) #返回状态值
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

model = Net(len(text.vocab.stoi), 100, 200, 256) #fix_length=200

if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    for x, y in trainloader:
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








