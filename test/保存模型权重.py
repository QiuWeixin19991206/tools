import torch
'''保存模型'''
PATH = 'my_model.pth'
torch.save(model.state_dict(), PATH)#保存模型权重
'''加载'''
new_model = Net()#重新初始化网络
new_model.load_state_dict(torch.load(PATH))#加载网络权重

if torch.cuda.is_available():
    new_model.to('cuda')

test_correct = 0
test_total = 0
new_model.eval()#预测模式
with torch.no_grad():
    for x, y in test_dl:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = new_model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)

epoch_test_acc = test_correct / test_total
print(epoch_test_acc)



'''保存效果最好的权重'''
model = Net()
if torch.cuda.is_available():
    model.to('cuda')

optim = torch.optim.Adam(model.parameters(), lr=0.001)
#保存效果最好的权重
import copy
best_model_wts = copy.deepcopy(model.state_dict())#深拷贝
best_acc = 0.0#用于计算最好的得分

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    #fit没复制过来
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,test_dl)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    #判断是否为最好的模型权重 则保存
    if epoch_test_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_wts)
model.eval()
# %% md
# 完整模型的保存和加载
# %%
PATH = './my_whole_model.pth'
# %%
torch.save(model, PATH)
# %%
new_model2 = torch.load(PATH)
new_model2.eval()
# %%
new_model2
# %% md
# 跨设备的模型保存和加载
# %% md
### GPU保存，CPU加载
# %%
PATH = './my_gpu_model_wts'
# %%
torch.save(model.state_dict(), PATH)
# %%
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))
# %% md
### 保存在GPU 上，在 GPU 上加载
# %%
PATH = './my_gpu_model2_wts'
# %%
torch.save(model.state_dict(), PATH)
# %%
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)
# %% md
### 保存 CPU 上，在 GPU 上加载
# %%
PATH = 'my_cpu_wts.pth'
# %%
torch.save(model.state_dict(), PATH)
# %%
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

