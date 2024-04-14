import torch
import torchvision
from torch import nn


class ResnetbasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)#会减少(k-1)，padding=1上下左右添加1个
        self.bn1 = nn.BatchNorm2d(out_channels)#因为bn相消了bias 所以上面这层不用bias减少运算

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)#会减少(k-1)，padding=1上下左右添加1个
        self.bn2 = nn.BatchNorm2d(out_channels)#因为bn相消了bias 所以上面这层不用bias减少运算

    def forward(self, x):
        residul = x
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))

        out = self.conv2(out)
        out = torch.relu(self.bn2(out))

        out += residul
        out = torch.relu(out)
        return out


