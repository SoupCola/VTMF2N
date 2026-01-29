'''
Model Factory: Dynamically builds the specified model based on configuration.
'''

import sys
import torch.nn as nn
import sys

from Models.SingleModality import *


def get_model(config, use_gpu):
    """
    Build and return the model instance based on configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
        use_gpu (bool): Whether to use GPU for computation.
    
    Returns:
        model (nn.Module): Instantiated model ready for training/inference.
        model_params (dict): Dictionary of actual model hyperparameters used.
    """
    # Extract common video/image parameters from config
    model_type = config['Model_name']
    img_width = config['Image_width']
    img_height = config['Image_height']
    video_length = config['video_length']
    scale_ratio = config['scale_ratio']
    
    # Compute resized dimensions after scaling
    resized_width = int(img_width * scale_ratio)
    resized_height = int(img_height * scale_ratio)

    model = None
    model_params = {}

    # ========================================================================
    # 1. CNN + LSTM Model
    # ========================================================================
    if model_type.lower() == 'cnn_lstm':
        # This is the benchmark model from slip detection literature.
        
        base_network = config['base_network']          # e.g., 'resnet18'
        pretrained = config['pretrained']              # bool: use ImageNet weights?
        frozen_weights = config['frozen_weights']      # bool: freeze backbone?
        dropout_cnn = config["CNN_drop"]               # CNN dropout rate
        dropout_lstm = config["LSTM_drop"]             # LSTM dropout rate

        # Fixed architecture parameters (as in original paper)
        rnn_input_size = 64      # Output dim of CNN per frame
        rnn_hidden_size = 64     # LSTM hidden state size
        rnn_num_layers = 1       # Number of LSTM layers
        num_classes = 2          # Binary classification: slip / no-slip

        # Instantiate the CNN-LSTM model
        model = CNN_LSTM_SlipDetector(
            base_network=base_network,
            pretrained=pretrained,
            frozen_weights=frozen_weights,
            dropout_CNN=dropout_cnn,
            rnn_input_size=rnn_input_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            num_classes=num_classes,
            use_gpu=use_gpu,
            video_length=video_length
        )

        # Record actual parameters used
        model_params = {
            'base_network': base_network,
            'rnn_input_size': rnn_input_size,
            'rnn_hidden_size': rnn_hidden_size,
            'rnn_num_layers': rnn_num_layers,
            'num_classes': num_classes,
            'dropout_CNN': dropout_cnn,
            'dropout_LSTM': dropout_lstm
        }

    # ========================================================================
    # 2. ViViT Model (Video Vision Transformer)
    # ========================================================================
    elif model_type.lower() == 'vivit':
        img_size = (resized_height, resized_width)
        patch_size = (12, 16)
        in_chans = 3
        num_classes = 2
        embed_dim = 256
        depth = 8
        num_heads = 16
        mlp_ratio = 4
        dropout = config['mlp_drop']
        attn_dropout = config['attn_drop']
        num_frames = video_length

        model = VIVIT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=dropout,
            attn_drop_rate=attn_dropout,
            drop_path_rate=dropout,
            num_frames=num_frames,
            dropout=0.,
            use_gpu=use_gpu
        )

        model_params = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_chans': in_chans,
            'num_cls': num_classes,
            'emb_dim': embed_dim,
            'depth': depth,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'dropout': dropout,
            'attn_drop_rate': attn_dropout,
            'num_frames': num_frames
        }

    # ========================================================================
    # 3. TimeSformer Model
    # ========================================================================
    elif model_type.lower() == 'timesformer':
        img_size = (resized_height, resized_width)
        patch_size = (12, 16)
        in_chans = 3
        num_classes = 2
        embed_dim = 256
        depth = 8
        num_heads = 16
        mlp_ratio = 4
        dropout = config['mlp_drop']
        attn_dropout = config['attn_drop']
        num_frames = video_length

        model = TimeSFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=dropout,
            attn_drop_rate=attn_dropout,
            drop_path_rate=dropout,
            hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            num_frames=num_frames,
            attention_type='divided_space_time',
            dropout=0.,
            use_gpu=use_gpu
        )

        model_params = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_chans': in_chans,
            'num_cls': num_classes,
            'emb_dim': embed_dim,
            'depth': depth,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'dropout': dropout,
            'attn_drop_rate': attn_dropout,
            'num_frames': num_frames
        }

    # ========================================================================
    # Unsupported Model Type
    # ========================================================================
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total parameters.")          # 总参数量
    print(f"{trainable_params:,} trainable parameters.")  # 可训练参数量

    return model, model_params


# =============================================================================
# Test Script for get_model
# =============================================================================
if __name__ == "__main__":
    import torch

    # Simulate a config dictionary (matching your training setup)
    test_config = {
        'Model_name': 'timeSformer_single',
        'Image_width': 320,
        'Image_height': 240,
        'video_length': 8,
        'scale_ratio': 1.0,
        'base_network': 'resnet18',      # Only ResNet variants supported
        'pretrained': False,
        'frozen_weights': False,
        'CNN_drop': 0.5,
        'LSTM_drop': 0.8,
        'mlp_drop': 0.1,                 # unused in basic_CNN, but required by config
        'attn_drop': 0.1
    }

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Build model using factory
    model, params = get_model(test_config, use_gpu=use_gpu)
    model = model.to(device)

    # Create dummy input: [Batch, Channels, Time, Height, Width]
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, test_config['video_length'], 240, 320).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Print shapes
    print("\n Test Passed!")
    print("Input shape :", dummy_input.shape)   # Expected: torch.Size([1, 3, 8, 240, 320])
    print("Output shape:", output.shape)        # Expected: torch.Size([1, 2])