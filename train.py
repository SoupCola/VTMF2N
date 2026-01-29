"""
Model Training Module  
Contains all functions needed to train the Vision-Tactile Multimodal Fusion Network (VTMF2N)  
"""

import os
import sys
import time
import yaml
import json
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# Import project-specific utilities  
from utils import data_loader
from utils import model_factory          # for multimodal models    
from utils import model_factory_single   # for single-modality models 
from utils import log_record             # logging utility           
from utils.plot_results import plot_loss_accuracy

# Check CUDA availability  
cuda_available = torch.cuda.is_available()


# Learning rate scheduler with warmup  
def lr_lambda(step, warmup_steps, init_lr, decay_factor):
    """Lambda function for learning rate scheduling with warmup and exponential decay  
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return decay_factor ** (step - warmup_steps)


# Main training function  
def train_net(params):
    """
    Execute full training pipeline based on configuration  
    """

    # ========================================================================
    # 1. Device Setup (CPU/GPU)  
    # ========================================================================
    use_gpu = False
    device = torch.device("cpu")

    if params['use_gpu'] == 1:
        print("GPU enabled in config: use_gpu=1")
        if cuda_available:
            print("CUDA available → Using GPU")
            device = torch.device("cuda:0")
            use_gpu = True
        else:
            print("CUDA not available → Falling back to CPU")
    else:
        print("GPU disabled in config → Using CPU")

    # ========================================================================
    # 2. Directory and Logging Setup  
    # ========================================================================
    exp_save_dir = None
    if 'save_dir' in params:
        model_save_dir = os.path.join(params['save_dir'], params['Model_name'])
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        # Create timestamped experiment folder  
        timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        exp_save_dir = os.path.join(model_save_dir, timestamp)
        os.mkdir(exp_save_dir)

        # Initialize log file  
        log_record.create_log(exp_save_dir)

        # Save config copy  
        yaml_path = os.path.join(exp_save_dir, 'config.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    # ========================================================================
    # 3. Reproducibility & Model Initialization  
    # ========================================================================
    if params.get('use_random_seed', 1) == 0:
        torch.manual_seed(params['seed'])

    # Build model based on modality  
    if params['Modality'] == "Combined":
        model, model_params = model_factory.get_model(params, use_gpu)
    elif params['Modality'] in ["Tactile", "Visual"]:
        model, model_params = model_factory_single.get_model(params, use_gpu)
    else:
        raise ValueError(f"Unsupported Modality: {params['Modality']}")

    # Save model architecture info  
    if exp_save_dir:
        model_params_path = os.path.join(exp_save_dir, 'model_params.json')
        with open(model_params_path, 'w') as f:
            json.dump(model_params, f)

    # Initialize weights if required  
    if params.get('skip_init_in_train', 1) == 0:
        model.init_weights()

    # Move model to device  
    if use_gpu:
        model = model.cuda()

    # ========================================================================
    # 4. Optimizer, Scheduler & Loss Function  
    # ========================================================================
    loss_fn = nn.CrossEntropyLoss()
    if use_gpu:
        loss_fn = loss_fn.cuda()

    # Configure optimizer with optional warmup  
    if params.get('adam_warmup', False):
        optimizer = optim.Adam(model.parameters(), lr=float(params['lr']))
        warmup_epochs = params['warmup_epochs']
        base_lr = float(params['lr'])
        decay_factor = params['decay_factor']
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_epochs, base_lr, decay_factor))
        print('Using Adam with warmup strategy')
        print(f'Base LR: {base_lr}, Warmup epochs: {warmup_epochs}')
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(params['lr']))
        scheduler = None

    # ========================================================================
    # 5. Data Loaders  
    # ========================================================================
    common_args = {
        'scale_ratio': params["scale_ratio"],
        'video_length': params["video_length"]
    }

    train_dataset = data_loader.Tactile_Vision_dataset(data_path=params['Train_data_dir'], **common_args)
    valid_dataset = data_loader.Tactile_Vision_dataset(data_path=params['Valid_data_dir'], **common_args)
    test_dataset = data_loader.Tactile_Vision_dataset(data_path=params['Test_data_dir'], **common_args)

    dataloader_kwargs = {
        'batch_size': params['batch_size'],
        'shuffle': False,
        'num_workers': params['num_workers']
    }

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    valid_loader = DataLoader(valid_dataset, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **dataloader_kwargs)

    # ========================================================================
    # 6. Forward Pass Test (Sanity Check)  
    # ========================================================================
    sample_batch = next(iter(train_loader))
    rgb_imgs, tactile_imgs, labels = sample_batch

    print("Running feed-forward sanity check:")
    if params['Modality'] == "Combined":
        output = model(rgb_imgs, tactile_imgs)
    elif params['Modality'] == "Visual":
        output = model(rgb_imgs)
    elif params['Modality'] == "Tactile":
        output = model(tactile_imgs)

    _, pred = torch.max(output.data, 1)
    print("Model output:", output)
    print("Prediction:", pred)
    print("Ground truth:", labels)
    print("✓ Feed-forward test passed!")

    # ========================================================================
    # 7. Training Loop  
    # ========================================================================
    train_loss_hist, train_acc_hist = [], []
    valid_loss_hist, valid_acc_hist = [], []
    best_test_acc = 0.0

    start_time = time.time()

    for epoch in range(params['epochs']):
        # --------------------------------------------------------------------
        # Training Phase  
        # --------------------------------------------------------------------
        model.train()
        train_total_loss = 0.0
        train_correct = 0
        train_total_samples = 0

        for batch_idx, (rgb, tactile, label) in enumerate(train_loader):
            model.zero_grad()

            # Forward pass  
            if params['Modality'] == "Combined":
                    logits = model(rgb, tactile)
            elif params['Modality'] == "Visual":
                logits = model(rgb)
            else:  # Tactile
                logits = model(tactile)

            if use_gpu:
                label = label.cuda()

            # Compute loss  
            loss = loss_fn(logits, label)

            # Backward & optimize  
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track metrics  
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == label).sum().item()
            train_total_loss += loss.item()
            train_total_samples += label.size(0)

        avg_train_loss = train_total_loss / train_total_samples
        avg_train_acc = train_correct / train_total_samples
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)

        # --------------------------------------------------------------------
        # Validation Phase  
        # --------------------------------------------------------------------
        model.eval()
        valid_total_loss = 0.0
        valid_correct = 0
        valid_total_samples = 0

        with torch.no_grad():
            for rgb, tactile, label in valid_loader:
                if params['Modality'] == "Combined":
                        logits = model(rgb, tactile)
                elif params['Modality'] == "Visual":
                    logits = model(rgb)
                else:
                    logits = model(tactile)

                if use_gpu:
                    label = label.cuda()

                loss = loss_fn(logits, label)

                _, predicted = torch.max(logits.data, 1)
                valid_correct += (predicted == label).sum().item()
                valid_total_loss += loss.item()
                valid_total_samples += label.size(0)

        avg_valid_loss = valid_total_loss / valid_total_samples
        avg_valid_acc = valid_correct / valid_total_samples
        valid_loss_hist.append(avg_valid_loss)
        valid_acc_hist.append(avg_valid_acc)

        # --------------------------------------------------------------------
        # Logging & Timing  
        # --------------------------------------------------------------------
        elapsed = time.time() - start_time
        avg_time_per_epoch = elapsed / (epoch + 1)
        avg_time_per_batch = avg_time_per_epoch / len(train_loader)
        eta = avg_time_per_epoch * params['epochs'] - elapsed

        if epoch % params.get('print_interval', 1) == 0:
            msg_train = f"[Epoch {epoch:3d}/{params['epochs']}] Train Loss: {avg_train_loss:.3f}, Acc: {avg_train_acc:.3f}"
            msg_valid = f"[Epoch {epoch:3d}/{params['epochs']}] Valid Loss: {avg_valid_loss:.3f}, Acc: {avg_valid_acc:.3f}"
            msg_time = f"Elapsed {elapsed:.2f}s, {avg_time_per_epoch:.2f}s/epoch, {avg_time_per_batch:.2f}s/batch, ETA {eta:.2f}s"

            print(msg_train)
            print(msg_valid)
            print(msg_time)

            if exp_save_dir:
                log_record.update_log(exp_save_dir, msg_train)
                log_record.update_log(exp_save_dir, msg_valid)
                log_record.update_log(exp_save_dir, msg_time)

        # --------------------------------------------------------------------
        # Optional Test Evaluation & Model Saving  
        # --------------------------------------------------------------------
        if params.get('test_eval', 0) == 1:
            model.eval()
            test_total_loss = 0.0
            test_correct = 0
            test_total_samples = 0

            with torch.no_grad():
                for rgb, tactile, label in test_loader:
                    if params['Modality'] == "Combined":
                        logits = model(rgb, tactile)
                    elif params['Modality'] == "Visual":
                        logits = model(rgb)
                    else:
                        logits = model(tactile)

                    if use_gpu:
                        label = label.cuda()

                    _, predicted = torch.max(logits.data, 1)
                    test_correct += (predicted == label).sum().item()
                    test_total_loss += loss.item()
                    test_total_samples += label.size(0)

            test_acc = test_correct / test_total_samples
            test_loss = test_total_loss / test_total_samples

            print(f"Current test accuracy: {test_acc:.4f}")

            # Save best model based on test accuracy  
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                msg_best = f"★ Model Improved! [Epoch {epoch}] Test Loss: {test_loss:.3f}, Acc: {test_acc:.3f}"
                print(msg_best)
                if exp_save_dir:
                    log_record.update_log(exp_save_dir, msg_best)
                    model_path = os.path.join(exp_save_dir, f"{params['Model_name']}_{epoch:05d}.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, model_path)

        print("")  # Empty line between epochs

    # ========================================================================
    # 8. Final Test Evaluation & Cleanup  
    # ========================================================================
    if params.get('test_eval', 0) == 1:
        model.eval()
        test_total_loss = 0.0
        test_correct = 0
        test_total_samples = 0

        with torch.no_grad():
            for rgb, tactile, label in test_loader:
                if params['Modality'] == "Combined":
                    logits = model(rgb, tactile)
                elif params['Modality'] == "Visual":
                    logits = model(rgb)
                else:
                    logits = model(tactile)

                if use_gpu:
                    label = label.cuda()

                loss = loss_fn(logits, label)

                _, predicted = torch.max(logits.data, 1)
                test_correct += (predicted == label).sum().item()
                test_total_loss += loss.item()
                test_total_samples += label.size(0)

        final_test_acc = test_correct / test_total_samples
        final_test_loss = test_total_loss / test_total_samples
        msg_final = f"Final Test Results → Loss: {final_test_loss:.3f}, Acc: {final_test_acc:.3f}"
        print(msg_final)
        if exp_save_dir:
            log_record.update_log(exp_save_dir, msg_final)

    # Save final model and plots  
    if exp_save_dir:
        last_model_path = os.path.join(exp_save_dir, f"{params['Model_name']}_last.pt")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, last_model_path)

        plot_loss_accuracy(
            train_loss_hist, valid_loss_hist,
            train_acc_hist, valid_acc_hist,
            exp_save_dir, colors=['blue']
        )