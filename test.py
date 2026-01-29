"""
Model Evaluation / Testing Script  
Evaluate a trained vision-tactile model on test data  
模型评估/测试脚本：在测试集上评估已训练的视觉-触觉模型
"""

import os
import sys
import yaml
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

# Import project utilities  
from utils import data_loader
from utils import model_factory          # for multimodal models      
from utils import model_factory_single   # for single-modality models

# Check if CUDA is available  
cuda_available = torch.cuda.is_available()


# Main testing function  
def test_net(config, model):
    """
    Evaluate the model on the test dataset  
    """

    # ========================================================================
    # 1. Device Setup (CPU/GPU)  
    # 1. 设备设置（CPU/GPU）
    # ========================================================================
    if cuda_available:
        print("CUDA available → Using GPU")
        device = torch.device("cuda:0")
    else:
        print("CUDA not available → Using CPU")
        device = torch.device("cpu")

    # ========================================================================
    # 2. Loss Function Setup  
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    if cuda_available:
        model = model.cuda()
        criterion = criterion.cuda()

    # ========================================================================
    # 3. Test Data Loader  
    # ========================================================================
    test_dataset = data_loader.Tactile_Vision_dataset(
        scale_ratio=config["scale_ratio"],
        video_length=config["video_length"],
        data_path=config['Test_data_dir']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,             
        num_workers=config['num_workers']
    )

    # ========================================================================
    # 4. Evaluation Loop  
    # ========================================================================
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()  # Set to evaluation mode  

    start_time_total = time.time()

    with torch.no_grad():  # Disable gradient computation  
        for rgb_imgs, tactile_imgs, labels in test_loader:
            batch_start = time.time()

            # Forward pass based on modality  
            if config['Modality'] == "Combined":
                output = model(rgb_imgs, tactile_imgs)
            elif config['Modality'] == "Visual":
                output = model(rgb_imgs)
            elif config['Modality'] == "Tactile":
                output = model(tactile_imgs)

            # Move labels to GPU if needed  
            if cuda_available:
                labels = labels.cuda()

            # Print per-batch inference time  
            print("One batch elapsed time: %f seconds" % (time.time() - batch_start))

            # Compute loss and accuracy  
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

    # Final metrics  
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    print('Test Results → Loss: %.3f, Accuracy: %.3f' % (avg_loss, avg_acc))
    print("Total evaluation time: %f seconds" % (time.time() - start_time_total))


# ========================================================================
# 5. Main Entry Point  
# ========================================================================
if __name__ == "__main__":
    # Hardcoded paths for model and config (as in original)  
    model_path = './runs/VTMF2N/basic_CNN_00000.pt'  # Best model
    yaml_config_path = './config.yaml'

    # Load YAML configuration  
    if os.path.exists(yaml_config_path):
        with open(yaml_config_path) as f:
            config_loaded = yaml.safe_load(f)
    else:
        print("Error: Config file does not exist!")
        sys.exit()

    # Build model architecture based on modality  
    if config_loaded['Modality'] == "Combined":
        model, _ = model_factory.get_model(config_loaded, cuda_available)
    elif config_loaded['Modality'] in ["Visual", "Tactile"]:
        model, _ = model_factory_single.get_model(config_loaded, cuda_available)
    else:
        raise ValueError(f"Unsupported Modality: {config_loaded['Modality']}")

    # Load trained weights  
    if cuda_available:
        checkpoint = torch.load(model_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    # Run evaluation  
    test_net(config_loaded, model)