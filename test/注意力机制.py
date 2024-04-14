'''Seq2Seq2有缺陷，注意力机制解决问题//自注意力机制（Google）'''
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# TORCHTEXT.DATASETS
# 所有数据集都是的子类torchtext.data.Dataset，它们继承自torch.utils.data.Dataset，
# 并且具有split和 iters实现的方法。
TEXT = torchtext.data.Field(lower=True, fix_length=200, batch_first=False)
LABEL = torchtext.data.Field(sequential=False)

# make splits for data
train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)

print(train.fields)
# 构建词表 vocab
# TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300),
#                 max_size=20000, min_freq=10)
TEXT.build_vocab(train, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train)

print(len(TEXT.vocab.freqs))

print(len(TEXT.vocab.stoi))   # unk & pad

print([(k, v) for k, v in TEXT.vocab.stoi.items() if v> 9999])


BATCHSIZE = 64

# make iterator for splits
train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=BATCHSIZE)
b = next(iter(train_iter))

hidden_size = 300
embeding_dim = 100


# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(20002, embeding_dim)  # 200*batch*100
        self.pos = PositionalEncoding(embeding_dim)#位置编码
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embeding_dim,
                                                        nhead=5)#embeding_dim得被nhead整除
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=6)
        self.fc1 = nn.Linear(200, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, inputs):
        x = self.em(inputs)#词嵌入
        x = self.pos(x)#位置嵌入
        x = self.transformer_encoder(x)#时间、batch、特征，提取特征
        x = x.permute(1, 0, 2)#batch，时间、特征
        x = torch.sum(x, dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
model = Net().to(device)
# %%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# %%
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for b in trainloader:
        x, y = b.text.to(device), b.label.to(device)
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
        for b in testloader:
            x, y = b.text.to(device), b.label.to(device)
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


# %%
epochs = 100
# %%
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_iter,
                                                                 test_iter)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
# %%
















