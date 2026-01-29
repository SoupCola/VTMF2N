'''
Multi-Modal Model Factory
'''

import sys
import torch.nn as nn


import sys

from Models import *

def get_model(config, use_gpu):
    """
    Build and return a multi-modal video classification model based on configuration.
    
    Args:
        config (dict): Configuration dictionary with model hyperparameters.
        use_gpu (bool): Whether to place the model on GPU.
    
    Returns:
        model (nn.Module): Instantiated model ready for training/inference.
        model_params (dict): Dictionary of actual hyperparameters used.
    """
    # Extract common configuration parameters
    model_type = config['Model_name']
    img_width = config['Image_width']
    img_height = config['Image_height']
    video_length = config['video_length']
    scale_ratio = config['scale_ratio']
    batch_size = config['batch_size']

    # Compute scaled image dimensions
    resized_width = int(img_width * scale_ratio)
    resized_height = int(img_height * scale_ratio)

    model = None
    model_params = {}

    # ========================================================================
    # 1. ViViT: Video Vision Transformer (Two-stream variant placeholder)
    # ========================================================================
    if model_type.lower() == 'vivit':
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
    # 2. Basic CNN + LSTM 
    # ========================================================================
    elif model_type.lower() == 'cnn_lstm':
        base_network = config['base_network']          # e.g., 'resnet18', 'resnet50'
        pretrained = config['pretrained']
        frozen_weights = config['frozen_weights']
        dropout_cnn = config["CNN_drop"]
        dropout_lstm = config["LSTM_drop"]

        # Fixed architecture settings (as per slip detection benchmark)
        rnn_input_size = 64      # Output dim per frame from CNN
        rnn_hidden_size = 64     # LSTM hidden state size
        rnn_num_layers = 1       # Number of LSTM layers
        num_classes = 2          # Binary: slip / no-slip

        model = cnn_lstm(
            base_network=base_network,
            pretrained=pretrained,
            frozen_weights=frozen_weights,
            dropout_CNN=dropout_cnn,
            dropout_LSTM=dropout_lstm,
            rnn_input_size=rnn_input_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            num_classes=num_classes,
            use_gpu=use_gpu,
            video_length=video_length
        )

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
    # 3. TimeSformer (Original implementation with space-time attention)
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
    # 4. VTMF2N
    # ========================================================================
    elif model_type.lower() == 'vtmf2n':
        base_network = config['base_network']
        pretrained = config['pretrained']
        frozen_weights = config['frozen_weights']
        dropout_cnn = config["CNN_drop"]
        num_classes = 2

        num_csca = config["num_csca"]    # Number of csca blocks


        model = VTMF2N(
            base_network=base_network,
            pretrained=pretrained,
            frozen_weights=frozen_weights,
            dropout_CNN=dropout_cnn,
            num_classes=num_classes,
            num_csca=num_csca,
            use_gpu=use_gpu,
            video_length=video_length
        )

        model_params = {
            'num_classes': num_classes,
            'dropout_CNN': dropout_cnn,
            'video_length': video_length,
            'num_csca': num_csca,
        }


    # ========================================================================
    # Unsupported model type
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
# Test Script: Validate model construction and forward pass
# =============================================================================
if __name__ == "__main__":
    import torch

    # Simulate a realistic config for multi-modal slip detection
    test_config = {
        'Model_name': 'VTMF2N',           # Try: 'VTMF2N', 'timeSformer', etc.
        'Image_width': 640,
        'Image_height': 480,
        'video_length': 14,
        'scale_ratio': 0.5,
        'batch_size': 1,
        'base_network': 'resnet18',          # Only ResNet variants are supported
        'pretrained': False,
        'frozen_weights': False,
        'CNN_drop': 0.5,
        'LSTM_drop': 0.8,
        'mlp_drop': 0.1,
        'attn_drop': 0.1,
        'num_csca': 2,
    }

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Build model using factory
    model, params = get_model(test_config, use_gpu=use_gpu)
    model = model.to(device)

    # Create dummy input: [Batch, Channels, Time, Height, Width]
    input_tensor = torch.randn(
        test_config['batch_size'],
        3,
        test_config['video_length'],
        test_config['Image_height'],
        test_config['Image_width']
    ).to(device)

    # Forward pass (no gradient computation)
    with torch.no_grad():
        output = model(input_tensor, input_tensor)

    # Print results
    print("\n Model test passed!")
    print("Output shape:", output.shape)        # Expected: torch.Size([2, 2])