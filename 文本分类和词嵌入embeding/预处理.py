import string
import numpy as np
import torch.nn

s = 'Life is not easy for any of us.We must work, and above all we must believe in ourselves.We must believe that each one of us is able to do some thing well.And that we must work until we succeed.'
#string.punctuation#表示标点符号
for c in string.punctuation:
    s = s.replace(c, ' ').lower()#变小写
print(s)

'''分词方式1'''
a = list(s)#得到每一个字符 然后就可以使用独热编码
print(a)
'''分词方法2'''
b = s.split()#得到每一个单词，词嵌入和独热
print(b)
'''分词方法3 n-gram 词向量'''
c = np.unique(s.split())
vocab = dict((word, index) for index, word in enumerate(c))
s = [vocab.get(w) for w in s.split()]#s映射成词表表示
print(vocab, len(vocab), s, len(s))
d = np.zeros((len(s), len(vocab)))#总数 和 类别

#1.变为独热编码
for index, i in enumerate(s):
    d[index, i] = 1
print(d)
#2.词嵌入表示
em = torch.nn.Embedding(len(vocab), 20)#映射为20的张量上
s_em = em(torch.LongTensor(s))
print(s_em.shape)



