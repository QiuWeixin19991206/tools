import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
#每多少步衰减 *0.9
exp_lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 50, 80], gamma=0.9)
#分段学习
exp_lr_scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#添加位置
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
exp_lr_scheduler.step()####################################
epoch_loss = running_loss / len(trainloader.dataset)
epoch_acc = correct / total










