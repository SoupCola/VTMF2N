import torch
import torchvision as tv
from torch import nn
import math
import os
import numpy as np
import torch.nn.functional as F

class C3D_VT(nn.Module):
    def __init__(self, num_layers=2, num_classes=2):
        super(C3D_VT, self).__init__()

        self.v_c3d = C3D()  # 输出[batchsize, 16]
        self.t_c3d = C3D()

        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, v_inputs, t_inputs, drop=None):
        v_out = self.v_c3d(v_inputs)
        t_out = self.t_c3d(t_inputs)  # 输出[batchsize, 512]

        if drop != None:
                for i in range(len(drop)):
                    if drop[i] == 1:
                        v_out[i,:] = 0.0
                    elif drop[i] == 2:
                        t_out[i,:] = 0.0

        vt_out = torch.cat((v_out, t_out), 1)  # [batchsize, 1024]
        out = self.relu(vt_out)
        out = self.fc(out)
        out = self.softmax(out)
        return v_out, t_out, out
    

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # T: 10 → 5, H: 120 → 60, W: 160 → 80

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # T: 5 → 5, H: 60 → 30, W: 80 → 40

        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # T: 5 → 5, H: 30 → 15, W: 40 → 20

        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # T: 5 → 5, H: 15 → 7, W: 20 → 10

        self.conv5 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # T: 5 → 5, H: 7 → 3, W: 10 → 5

        self.conv6 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool6 = nn.AdaptiveMaxPool3d((5, 12, 16))  # 确保最终输出为 (5, 12, 16)

        self.fc6 = nn.Linear(64 * 5 * 12 * 16, 4096)  # 根据卷积后的输出形状调整输入尺寸
        self.fc7 = nn.Linear(4096, 2048)
        self.fc8 = nn.Linear(2048, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        x = self.pool6(self.conv6(x))
        x = x.view(x.size(0), -1)  # 将输出展平为 (batch_size, 64 * 5 * 12 * 16)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 测试部分
if __name__ == '__main__':
    use_gpu = True
    cuda_avail = torch.cuda.is_available()

    # 创建测试输入数据
    v_input = torch.rand(2, 3, 10, 120, 160) 
    t_input = torch.rand(2, 3, 10, 120, 160) 

    # 检测是否可用 GPU
    if use_gpu and cuda_avail:
        use_gpu = True
        v_input = v_input.to('cuda')
        t_input = t_input.to('cuda')
    else:
        use_gpu = False

    # 初始化模型
    model = C3D_VT(num_layers=2, num_classes=2)

    # 将模型移动到 GPU（如果可用）
    if use_gpu:
        model = model.to('cuda')

    # 设置模型为评估模式
    model.eval()

    # 前向推理
    with torch.no_grad():
        output = model(v_input, t_input)

    # 打印输出
    print("Model outputs shape:", output[-1].shape)
    print("Model output probabilities:", output)
