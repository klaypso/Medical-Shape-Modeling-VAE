
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

def Normalization(norm_type, out_channels,num_group=1):
    if norm_type==1:
        return nn.InstanceNorm3d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm3d(out_channels,momentum=0.1)
    elif norm_type==3:
        return GSNorm3d(out_channels,num_group=num_group)

class GSNorm3d(torch.nn.Module):
    def __init__(self, out_ch, num_group=1):
        super().__init__()
        self.out_ch = out_ch
        self.num_group=num_group
        #self.activation = nn.ReLU()
    def forward(self, x):
        interval = self.out_ch//self.num_group
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            #dominator = torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)
            #dominator = dominator + (dominator<0.001)*1
            tensors.append(x[:,start_index:start_index+interval,...]/(torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)+0.0001))
            start_index = start_index+interval
        
        return torch.cat(tuple(tensors),dim=1)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1), 
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Up_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch,num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='trilinear'),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Down_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Conv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1, activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, num_group=1,activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

            

class GSConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, num_group=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, if_sub=None,trainable=True):
        super(GSConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
        self.num_group = num_group
        self.interval = self.in_channels//self.num_group
    def forward(self, x):
        