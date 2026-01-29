# -*- coding: utf-8 -*-
import torch
import torchvision as tv
from torch import nn
import math
import os
import numpy as np
import torch.nn.functional as F 
from .backbone import resnet18
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:  # 非因果卷积
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:  # 因果卷积
            return x[:, :, :-self.chomp_size].contiguous()

class resnet_tcn(nn.Module):
    def __init__(self, input_size=64, num_channels=[64, 64, 64], kernel_size=[3, 3, 3], dropout=0.2, modality=None):
        super(resnet_tcn, self).__init__()
        self.cnn = resnet18(modality=modality)
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(512 * 8 * 10, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            cnn_output = self.cnn(inputs[:, t:t+1, :, :, :])
            cnn_output = self.dropout(cnn_output)
            cnn_output = cnn_output.view(cnn_output.size(0), -1)
            cnn_output = self.relu(self.fc1(cnn_output))
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc2(cnn_output))
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)  
        out = self.tcn(x)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # input_size=1,num_channels = [25 25 25 25 25 25 25 25],kernel_size=7,dropout=0.05
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1.shape)  # [8, 64, 13]
        # print(y1[:, :, -1].shape)  # [8, 64]
        # o = self.linear(y1[:, :, -1])
        return out
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """
                :param n_inputs: int, 输入通道数或者特征数
                :param n_outputs: int, 输出通道数或者特征数
                :param kernel_size: int, 卷积核尺寸
                :param stride: int, 步长, 在TCN固定为1
                :param dilation: int, 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系. 
                                        例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
                                              如果这个Residual block在第2层, dilation = 2**1 = 2;
                                              如果这个Residual block在第3层, dilation = 2**2 = 4;
                                              如果这个Residual block在第4层, dilation = 2**3 = 8 ......
                :param padding: int, 填充系数. 与kernel_size和dilation有关. 
                :param dropout: float, dropout比率
                """

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, False)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding, False)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print(x.shape)  # [8, 64, 5]
        out = self.net(x)  # input (N, C, L) L表示时间序列，这里是将28*28的图拆成784*1，每个时间序列的特征为1，共784个时间序列
        #print(out.shape)  # [8, 64, 5]
        res = x if self.downsample is None else self.downsample(x)
        #print(res.shape)
        out = out + res
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        """
                :param num_inputs: int,  输入通道数或者特征数
                :param num_channels: list, 每层的hidden_channel数. 例如[5,12,3], 代表有3个block, 
                                        block1的输出channel数量为5; 
                                        block2的输出channel数量为12;
                                        block3的输出channel数量为3.
                :param kernel_size: int, 卷积核尺寸
                :param dropout: float, drop_out比率
                """
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            k = kernel_size[i]
            layers += [TemporalBlock(in_channels, out_channels, k, stride=1, dilation=dilation_size,
                                     padding=(k-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MultiscaleTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(MultiscaleTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in kernel_size]
            layers += [MultiscaleTemporalConvNetBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class MultiscaleTemporalConvNetBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2):
        super(MultiscaleTemporalConvNetBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"
        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)  # setattr(object,name,value)设置属性值，用来存放单个卷积层
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # final relu
        self.relu = nn.ReLU()

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))  
            outputs.append(branch_convs(x))  # [8,32,5]
        out0 = torch.cat(outputs, 1)  
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out1 + res)


class SingleBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(SingleBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding, False)
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        return out

class MS_TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(MS_TCN, self).__init__()
        self.tcn = MultiscaleTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        return out

class cnn_mstcn(nn.Module):
    def __init__(self, num_classes=2):
        super(cnn_mstcn, self).__init__()
        input_size = 64
        num_channels = [64, 64]
        kernel_size = [5, 5]
        Mnum_channels = [64, 64, 64]
        Mkernel_size = [3, 3]
        dropout = 0.2

        self.v_resnet_tcn = resnet_tcn(input_size, num_channels, kernel_size, dropout, modality="visual")
        self.t_resnet_tcn = resnet_tcn(input_size, num_channels, kernel_size, dropout, modality="tactile")
        self.tcn = MS_TCN(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, v_inputs, t_inputs, drop=None):
        v_out = self.v_resnet_tcn(v_inputs)
        t_out = self.t_resnet_tcn(t_inputs)

        if drop != None:
            for i in range(len(drop)):
                if drop[i] == 1:
                    v_out[i,:] = 0.0
                elif drop[i] == 2:
                    t_out[i,:] = 0.0

        vt_output = torch.cat((v_out, t_out), 1)
        out = self.tcn(vt_output)
        out = out[:, :, -1]
        out = self.relu(out)
        out = self.fc(out)
        return v_out, t_out, out
    
    
if __name__ == "__main__":
    batch_size = 1
    video_length = 2  
    num_classes = 2
    
    model = cnn_mstcn(num_classes=num_classes)
    
    v_inputs = torch.randn(batch_size, video_length, 3, 240, 320)
    t_inputs = torch.randn(batch_size, video_length, 3, 240, 320)
    
    if torch.cuda.is_available():
        model = model.cuda()
        v_inputs = v_inputs.cuda()
        t_inputs = t_inputs.cuda()
    
    outputs = model(v_inputs, t_inputs)
    
    print("Model outputs shape:", outputs[-1].shape)

