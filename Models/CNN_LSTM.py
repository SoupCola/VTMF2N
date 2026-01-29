# -*- coding: utf-8 -*-
import torch
import torchvision as tv
from torch import nn
import math
import os
import numpy as np
import torch.nn.functional as F 


from .backbone import resnet18

class cnn_lstm(nn.Module):
    def __init__(self, num_layers=2, num_classes=2):
        super(cnn_lstm, self).__init__()
        self.cnn1 = resnet18(modality='visual')
        self.cnn2 = resnet18(modality='tactile')
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(512*8*10, 1024)  
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(512*8*10, 1024)
        self.fc4 = nn.Linear(1024, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, v_inputs, t_inputs, drop=None):
        cnn_output_list = list()
        for t in range(v_inputs.size(1)):
            v_cnn_output = self.cnn1(v_inputs[:, t:t+1, :, :, :]) 
            t_cnn_output = self.cnn2(t_inputs[:, t:t+1, :, :, :]) 
            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = v_cnn_output.view(v_cnn_output.size(0), -1)
            v_cnn_output = self.relu(self.fc1(v_cnn_output))  
            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = self.relu(self.fc2(v_cnn_output))

            t_cnn_output = self.dropout(t_cnn_output)
            t_cnn_output = t_cnn_output.view(t_cnn_output.size(0), -1)
            t_cnn_output = self.relu(self.fc3(t_cnn_output))
            t_cnn_output = self.dropout(t_cnn_output)
            t_cnn_output = self.relu(self.fc4(t_cnn_output))

            cnn_output = torch.cat((v_cnn_output, t_cnn_output), 1)
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=1)
        out, hidden = self.lstm(x) 
        x = out[:, -1, :] 
        x = self.relu(x)
        x = self.fc5(x) 
        return v_inputs, t_inputs, x

     
if __name__ == "__main__":
    batch_size = 3
    video_length = 2  
    num_classes = 2
    
    model = cnn_lstm(num_layers=2, num_classes=num_classes)
    
    v_inputs = torch.randn(batch_size, video_length, 3, 240, 320)
    t_inputs = torch.randn(batch_size, video_length, 3, 240, 320)
    
    if torch.cuda.is_available():
        model = model.cuda()
        v_inputs = v_inputs.cuda()
        t_inputs = t_inputs.cuda()
    
    outputs = model(v_inputs, t_inputs)
    
    print("Model outputs shape:", outputs[-1].shape)

