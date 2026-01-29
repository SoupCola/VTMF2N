'''
Neural Network Architecture for Multimodal Slip Detection (RGB-only version)
多模态滑动检测网络架构（此处为单流 RGB 版本）
'''

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from timm.models.layers import trunc_normal_

# Set device: use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. Feature Extractor Backbone (Only ResNet variants)
# 特征提取主干网络（仅支持 ResNet 系列）
# =============================================================================
class FeatureExtractor(nn.Module):
    """
    Extracts visual features using a pretrained ResNet backbone and maps to a fixed-size vector.
    使用预训练 ResNet 主干提取视觉特征，并映射为固定维度向量。
    Output: [B, rnn_input_size]
    """
    def __init__(self, 
                 base_network='resnet18', 
                 pretrained=False, 
                 frozen_weights=False, 
                 dropout=0.5, 
                 rnn_input_size=64):
        super(FeatureExtractor, self).__init__()
        
        # Load specified ResNet model
        if base_network == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            final_in_features = 512
        elif base_network == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained)
            final_in_features = 512
        elif base_network == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            final_in_features = 2048
        else:
            raise ValueError("Only resnet18, resnet34, and resnet50 are supported.")

        # Replace original fc layer with an intermediate 4096-dim layer
        # 替换原始全连接层为中间 4096 维层（保持与原代码一致）
        self.backbone.fc = nn.Linear(final_in_features, 4096)

        # Freeze backbone weights if required
        # 若 frozen_weights=True，则冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = not frozen_weights

        # Final projection to RNN input size (e.g., 64)
        # 最终投影到 RNN 输入维度（如 64）
        self.projection = nn.Sequential(
            nn.Linear(4096, rnn_input_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, image):
        # image: [B, 3, H, W]
        features = self.backbone(image)          # [B, 4096]
        projected = self.projection(features)    # [B, rnn_input_size]
        output = self.dropout(projected)         # [B, rnn_input_size]
        return output


# =============================================================================
# 2. RNN Classifier (LSTM-based)
# RNN 分类器（基于 LSTM）
# =============================================================================
class LSTMClassifier(nn.Module):
    """
    Processes a sequence of feature vectors using LSTM and predicts class logits.
    使用 LSTM 处理特征序列并输出类别 logits。
    Input:  [B, T, D]  (Batch, Time steps, Feature dim)
    Output: [B, num_classes]
    """
    def __init__(self, 
                 input_size=64, 
                 hidden_size=64, 
                 num_layers=2, 
                 num_classes=2, 
                 use_gpu=False, 
                 dropout=0.8):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        # LSTM layer (batch_first=True: input shape = [B, T, D])
        # LSTM 层（batch_first=True：输入形状为 [B, T, D]）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # dropout only between layers
        )

        # Final classification head
        # 最终分类头
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence):
        # sequence: [B, T, D]
        batch_size = sequence.size(0)

        # Initialize hidden and cell states to zero
        # 初始化隐藏状态和细胞状态为零
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.use_gpu:
            h0, c0 = h0.to(device), c0.to(device)

        # Forward through LSTM
        # 前向传播通过 LSTM
        lstm_out, _ = self.lstm(sequence, (h0, c0))  # lstm_out: [B, T, hidden_size]

        # Use the last time step for prediction
        # 使用最后一个时间步进行预测
        last_output = lstm_out[:, -1, :]             # [B, hidden_size]
        logits = self.classifier(last_output)        # [B, num_classes]

        return logits


# =============================================================================
# 3. Full End-to-End Model
# 完整端到端模型
# =============================================================================
class CNN_LSTM_SlipDetector(nn.Module):
    """
    End-to-end slip detection model: CNN (frame-wise) + LSTM (temporal modeling).
    端到端滑动检测模型：CNN（逐帧） + LSTM（时序建模）。
    Input:  [B, 3, T, H, W]  (RGB video clip)
    Output: [B, 2]           (logits for "slip" / "no-slip")
    """
    def __init__(self,
                 base_network='resnet18',
                 pretrained=False,
                 frozen_weights=False,
                 dropout_CNN=0.5,
                 rnn_input_size=64,
                 rnn_hidden_size=64,
                 rnn_num_layers=2,
                 num_classes=2,
                 use_gpu=False,
                 video_length=8):
        super(CNN_LSTM_SlipDetector, self).__init__()
        self.use_gpu = use_gpu
        self.video_length = video_length

        # Frame-level feature extractor
        # 帧级特征提取器
        self.cnn = FeatureExtractor(
            base_network=base_network,
            pretrained=pretrained,
            frozen_weights=frozen_weights,
            dropout=dropout_CNN,
            rnn_input_size=rnn_input_size
        )

        # Temporal classifier
        # 时序分类器
        self.lstm = LSTMClassifier(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=num_classes,
            use_gpu=use_gpu,
            dropout=0.8
        )

    def forward(self, video_clip):
        """
        Args:
            video_clip: Tensor of shape [B, 3, T, H, W]
        Returns:
            logits: Tensor of shape [B, 2]
        """
        batch_size = video_clip.shape[0]
        frame_features = []

        # Process each frame through CNN
        # 逐帧通过 CNN 提取特征
        for t in range(self.video_length):
            frame = video_clip[:, :, t, :, :]  # [B, 3, H, W]
            if self.use_gpu:
                frame = frame.to(device)
            feat = self.cnn(frame)             # [B, rnn_input_size]
            frame_features.append(feat.unsqueeze(1))  # [B, 1, D]

        # Concatenate over time dimension
        # 在时间维度上拼接
        sequence = torch.cat(frame_features, dim=1)  # [B, T, D]

        # Classify using LSTM
        # 使用 LSTM 进行分类
        logits = self.lstm(sequence)  # [B, 2]
        return logits

    def init_weights(self):
        """Initialize linear layers with truncated normal distribution.
           使用截断正态分布初始化线性层"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)


# =============================================================================
# 4. Test Script
# 测试脚本
# =============================================================================
if __name__ == "__main__":
    # Create dummy input: [Batch, Channels, Time, Height, Width]
    # 创建虚拟输入：[批次, 通道, 时间帧数, 高, 宽]
    batch_size = 2
    video_length = 8
    dummy_video = torch.randn(batch_size, 3, video_length, 240, 320)

    # Initialize model (using ResNet18)
    # 初始化模型（使用 ResNet18）
    model = CNN_LSTM_SlipDetector(
        base_network='resnet18',
        pretrained=False,
        use_gpu=(device.type == 'cuda'),
        video_length=video_length
    ).to(device)

    # Forward pass
    # 前向传播
    output = model(dummy_video)
    
    # Print shapes
    # 打印张量形状
    print("Input shape:", dummy_video.shape)      # Expected: torch.Size([2, 3, 8, 240, 320])
    print("Output shape:", output.shape)          # Expected: torch.Size([2, 2])

    # Optional: print model parameters
    # 可选：打印模型参数名称与尺寸
    # for name, param in model.named_parameters():
    #     print(name, ":", param.shape)