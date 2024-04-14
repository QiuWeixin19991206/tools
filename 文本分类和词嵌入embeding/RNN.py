import torch.nn
import torchtext
import numpy as np
import torch. nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe


text = torchtext.legacy.data.Field(lower=True, fix_length=200, batch_first=True)
label = torchtext.legacy.data.Field(sequential=False)
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
        ht = torch.zeros(bz, hidden)
        for word in inputs:
            ht = self.rnn(word, ht)#ht上一个状态输出
        return ht


torch.nn.RNNCell()













