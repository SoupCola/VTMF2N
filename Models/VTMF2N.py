import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.nn.utils import weight_norm
from timm.models.layers import trunc_normal_

# Set device: use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Custom Normalization Layer: Group Normalization (GN)
# =============================================================================
class GN(nn.Module):
    """Group Normalization with learnable affine parameters.
       """
    def __init__(self, groups: int, channels: int, eps: float = 1e-5, affine: bool = True):
        super(GN, self).__init__()
        assert channels % groups == 0, 'channels must be divisible by groups'
        self.groups = groups          # Number of groups / 分组数
        self.channels = channels      # Total input channels / 输入通道总数
        self.eps = eps                # Small value to avoid division by zero / 防止除零的小常数
        self.affine = affine          # Whether to apply learnable scale & shift / 是否使用可学习缩放和偏移
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))   # Learnable scale / 可学习缩放因子
            self.shift = nn.Parameter(torch.zeros(channels))  # Learnable bias / 可学习偏置

    def forward(self, x: torch.Tensor):
        x_shape = x.shape  # [B, C, H, W]
        B = x_shape[0]
        assert self.channels == x.shape[1], "Channel mismatch"

        # Reshape to [B, G, -1] for group-wise normalization
        x = x.view(B, self.groups, -1)
        mean = x.mean(dim=-1, keepdim=True)          # [B, G, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # Use direct variance for stability
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm.view(B, self.channels, -1)
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)


# =============================================================================
# 2. Parallel Module Utilities
# =============================================================================
class ModuleParallel(nn.Module):
    """Apply the same module to multiple inputs in parallel.
       """
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution for channel adjustment.
       """
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class BatchNorm2dParallel(nn.Module):
    """Parallel BatchNorm2d for multi-stream inputs.
       """
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, f'bn_{i}', nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, f'bn_{i}')(x) for i, x in enumerate(x_parallel)]


# =============================================================================
# 3. Shift & Shuffle Operation (for cross-stream interaction)
# =============================================================================
class ShiftShuffle(nn.Module):
    """Shift and shuffle channels between two feature streams.
       """
    def __init__(self, reverse=False):
        super(ShiftShuffle, self).__init__()
        self.pos = [[-1, 0], [0, -1], [0, 1], [1, 0]]  # Up, Left, Right, Down
        if reverse:
            self.pos = self.pos[::-1]

    def forward(self, x):
        if len(x) != 2:
            return x, [0, 0]

        shift_group = x[0].shape[1] // 5
        shuffle_channel = shift_group * 4

        # Swap non-shifted part
        x1a, x1b = x[0][:, :shuffle_channel], x[0][:, shuffle_channel:]
        x2a, x2b = x[1][:, :shuffle_channel], x[1][:, shuffle_channel:]
        shuffled = [torch.cat([x2a, x1b], dim=1), torch.cat([x1a, x2b], dim=1)]

        h, w = x1a.shape[-2:]
        pad = (1, 1, 1, 1)
        x1_shifted, x2_shifted = [], []

        for idx in range(4):  # 4 directions
            posh, posw = self.pos[idx][0] + 1, self.pos[idx][1] + 1
            ch_start, ch_end = idx * shift_group, (idx + 1) * shift_group
            x1_shifted.append(F.pad(x1a[:, ch_start:ch_end], pad)[:, :, posh:h+posh, posw:w+posw])
            x2_shifted.append(F.pad(x2a[:, ch_start:ch_end], pad)[:, :, posh:h+posh, posw:w+posw])

        # Append zero-padded remainder
        x1_shifted.append(torch.zeros_like(x1b))
        x2_shifted.append(torch.zeros_like(x2b))

        shifted = [torch.cat(x1_shifted, dim=1), torch.cat(x2_shifted, dim=1)]
        return shuffled, shifted


# =============================================================================
# 4. Feature Extractor Backbone
# =============================================================================
class BasicFeatureExtractor(nn.Module):
    """ResNet18-based feature extractor with adaptive pooling.
       基于 ResNet18 的特征提取器，带自适应池化"""
    def __init__(self, pretrained=False, frozen_weights=False, dropout=0.5):
        super(BasicFeatureExtractor, self).__init__()
        # Load ResNet18 backbone
        backbone = resnet18(pretrained=pretrained)
        # Remove final avgpool and fc layers
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        # Add adaptive average pooling to fix spatial size to (8,8)
        self.features.add_module('AdaptiveAvgPool', nn.AdaptiveAvgPool2d((8, 8)))

        # Freeze weights if required
        for param in self.features.parameters():
            param.requires_grad = not frozen_weights

        # Post-processing layers
        self.dropout_ = nn.Dropout(dropout)
        self.conv_reduce = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)  # Channel reduction
        self.conv_rgb = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_tactile = nn.Conv2d(64, 64, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, rgb_img, tactile_img):
        # Extract features from both modalities
        rgb_feat = self.features(rgb_img)        # [B, 512, 8, 8]
        tactile_feat = self.features(tactile_img)

        # Reduce channel dimension
        rgb_feat = self.conv_reduce(rgb_feat)    # [B, 64, 8, 8]
        tactile_feat = self.conv_reduce(tactile_feat)

        # Modality-specific refinement
        rgb_feat = self.relu(self.conv_rgb(rgb_feat))
        tactile_feat = self.relu(self.conv_tactile(tactile_feat))

        rgb_feat = self.dropout_(rgb_feat)
        tactile_feat = self.dropout_(tactile_feat)

        return rgb_feat, tactile_feat

# =============================================================================
# 6. Channel Shuffle and Cross-Attention Module
# =============================================================================
class CSCA(nn.Module):
    def __init__(self, inplanes=160, planes=160, d_model=64, nhead=8, dropout=0.1, video_length=8):
        super(CSCA, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.shiftshuffle1 = ShiftShuffle(reverse=False)
        self.shiftshuffle2 = ShiftShuffle(reverse=True)

        inplanes = inplanes // 2
        planes = planes // 2
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel=2)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.gn = GN(groups=8, channels=64 * video_length)
        self.relu1 = nn.ReLU()

    def forward(self, rgb_feat, tactile_feat):
        dim = rgb_feat.size(3)  # spatial size (8)

        # Split each stream into two halves
        rgb_parts = torch.chunk(rgb_feat, 2, dim=1)
        tactile_parts = torch.chunk(tactile_feat, 2, dim=1)

        # Apply shift-shuffle
        rgb_shuffled, rgb_shift = self.shiftshuffle1(rgb_parts)
        tactile_shuffled, tactile_shift = self.shiftshuffle1(tactile_parts)

        # Process through shared conv + BN + ReLU
        rgb_processed = self.conv1(rgb_shuffled)
        tactile_processed = self.conv1(tactile_shuffled)
        rgb_processed = self.bn1(rgb_processed)
        tactile_processed = self.bn1(tactile_processed)
        rgb_processed = self.relu(rgb_processed)
        tactile_processed = self.relu(tactile_processed)

        # Add shifted features back
        rgb_final = [rgb_processed[0] + rgb_shift[1], rgb_processed[1] + rgb_shift[0]]
        tactile_final = [tactile_processed[0] + tactile_shift[1], tactile_processed[1] + tactile_shift[0]]

        # Reverse shift-shuffle
        rgb_final, _ = self.shiftshuffle2(rgb_final)
        tactile_final, _ = self.shiftshuffle2(tactile_final)

        # Recombine and normalize
        rgb_out = torch.cat(rgb_final, dim=1)
        tactile_out = torch.cat(tactile_final, dim=1)
        rgb_out = self.gn(rgb_out)
        tactile_out = self.gn(tactile_out)
        rgb_out = self.relu1(rgb_out)
        tactile_out = self.relu1(tactile_out)

        # Flatten for attention
        B, C, H, W = rgb_out.shape
        rgb_flat = rgb_out.reshape(B, C, -1)      # [B, C, H*W]
        tactile_flat = tactile_out.reshape(B, C, -1)

        # Cross-attention between streams
        attn_rgb, _ = self.multihead_attn(rgb_flat, tactile_flat, tactile_flat)
        attn_tactile, _ = self.multihead_attn(tactile_flat, rgb_flat, rgb_flat)

        # Residual connection
        rgb_enhanced = rgb_flat + attn_rgb
        tactile_enhanced = tactile_flat + attn_tactile

        # Reshape back to spatial format
        rgb_enhanced = rgb_enhanced.reshape(B, C, H, W)
        tactile_enhanced = tactile_enhanced.reshape(B, C, H, W)

        return rgb_enhanced, tactile_enhanced


# =============================================================================
# 7. Temporal Convolutional Network (TCN)
# =============================================================================
class Chomp1d(nn.Module):
    """Remove extra padding introduced by causal convolution.
       """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block for TCN with dilated causal convolutions.
       """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacked Temporal Blocks forming a TCN.
       """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """TCN wrapper for sequence modeling.
       """
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        return self.tcn(x)


# =============================================================================
# 8. Full Model: VTMF2N 
# =============================================================================
class VTMF2N(nn.Module):
    """Multimodal slip detection network using RGB + Tactile videos.
       """
    def __init__(self,
                 base_network='resnet18',
                 pretrained=False,
                 use_sa=False,
                 num_csca=2,
                 num_classes=2,
                 use_gpu=True,
                 frozen_weights=False,
                 dropout_CNN=0.5,
                 video_length=8,
                 dropout_ca=0.5):
        super(VTMF2N, self).__init__()
        assert base_network == 'resnet18', "Only resnet18 is supported now."

        self.feature_extractor = BasicFeatureExtractor(
            pretrained=pretrained,
            frozen_weights=frozen_weights,
            dropout=dropout_CNN
        )
        self.video_length = video_length
        self.use_gpu = use_gpu
        self.use_sa = use_sa
        self.num_csca = num_csca

        # csca module for cross-modal interaction
        self.csca = CSCA(
            inplanes=video_length * 64,
            planes=video_length * 64,
            dropout=dropout_ca,
            video_length=video_length
        )

        # Channel compression after csca
        self.compress_conv = nn.Conv2d(video_length * 64, video_length * 64 // 2, kernel_size=1)

        # TCN-based temporal modeling
        input_size = 64 * 8 * 8  # spatial features per frame
        num_channels = [512, 256, 128, 64]
        kernel_size = 5
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=0.2)
        self.final_relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, rgb_video, tactile_video):
        """
        Args:
            rgb_video: [B, 3, T, H, W]
            tactile_video: [B, 3, T, H, W]
        Returns:
            logits: [B, 2]
        """
        B, _, T, H, W = rgb_video.shape
        rgb_features_list = []
        tactile_features_list = []

        # Extract frame-wise features
        for t in range(self.video_length):
            rgb_frame = rgb_video[:, :, t, :, :]
            tactile_frame = tactile_video[:, :, t, :, :]
            if self.use_gpu:
                rgb_frame = rgb_frame.to(device)
                tactile_frame = tactile_frame.to(device)
            rgb_feat, tactile_feat = self.feature_extractor(rgb_frame, tactile_frame)
            rgb_features_list.append(rgb_feat.unsqueeze(1))      # [B, 1, 64, 8, 8]
            tactile_features_list.append(tactile_feat.unsqueeze(1))

        # Concatenate over time: [B, T, 64, 8, 8] → [B, T*64, 8, 8]
        rgb_seq = torch.cat(rgb_features_list, dim=1).view(B, -1, 8, 8)
        tactile_seq = torch.cat(tactile_features_list, dim=1).view(B, -1, 8, 8)

        # Apply csca blocks iteratively
        rgb_out, tactile_out = rgb_seq, tactile_seq
        for _ in range(self.num_csca):
            rgb_out, tactile_out = self.csca(rgb_out, tactile_out)

        # Compress channels
        rgb_compressed = self.compress_conv(rgb_out)
        tactile_compressed = self.compress_conv(tactile_out)

        # Fuse modalities
        fused = torch.cat([rgb_compressed, tactile_compressed], dim=1)  # [B, T*64, 8, 8]

        # Reshape for TCN: [B, C, H, W] → [B, C*H*W, T]
        fused_flat = fused.view(B, -1, self.video_length)  # [B, 4096, T]
        tcn_out = self.tcn(fused_flat)                     # [B, 64, T]
        final_feat = tcn_out[:, :, -1]                     # Last timestep: [B, 64]
        final_feat = self.final_relu(final_feat)
        output = self.classifier(final_feat)               # [B, 2]

        return output

    def init_weights(self):
        """Initialize linear layers with truncated normal distribution.
           """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)


# =============================================================================
# 9. Test Script
# =============================================================================
if __name__ == '__main__':
    video_length = 8
    batch_size = 1

    # Create dummy input: [B, 3, T, H, W]
    dummy_rgb = torch.randn(batch_size, 3, video_length, 240, 320)
    dummy_tactile = torch.randn(batch_size, 3, video_length, 240, 320)

    # Initialize model
    model = VTMF2N(
        base_network='resnet18',
        pretrained=False,
        video_length=video_length
    ).to(device)

    # Forward pass
    start_time = time.time()
    output = model(dummy_rgb, dummy_tactile)
    elapsed = time.time() - start_time

    print(f"Inference time: {elapsed:.4f} seconds")
    print(f"Output shape: {output.shape}")  # Expected: [1, 2]