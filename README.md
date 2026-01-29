# VTMF2N: Towards Accurate Visual-Tactile Slip Detection via Multi-modal Feature Fusion in Robotic Grasping

This repository contains the official implementation of the paper:

**VTMF2N: Towards Accurate Visual-Tactile Slip Detection via Multi-modal Feature Fusion in Robotic Grasping**

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── training/
│   ├── object1/
│   ├── object2/
│   └── ...
├── validation/
│   ├── object1/
│   ├── object2/
│   └── ...
└── testing/
    ├── object1/
    ├── object2/
    └── ...
```

Each object subdirectory should contain:
- `external_*.jpg`: External camera images (visual modality)
- `gelsight_*.jpg`: Tactile sensor images (tactile modality)
- `objectX_result.dat`: Labels and metadata file

## Configuration

Adjust the [config.yaml](./config.yaml) file according to your setup:

- `Train_data_dir`, `Valid_data_dir`, `Test_data_dir`: Path to your datasets
- `Model_name`: Select model ('VTMF2N', 'CNN_LSTM', 'ViViT', etc.)
- `batch_size`: Adjust according to your GPU memory
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `num_csca`: Number of CSCA blocks in VTMF2N (2 recommended)

## Training

To train the VTMF2N model:

```bash
python train.py
```

The training script will:
- Load the configuration from [config.yaml](./config.yaml)
- Create timestamped experiment directory in [./runs](./runs)
- Save model checkpoints and logs
- Generate loss and accuracy plots

## Testing

To evaluate a trained model:

```bash
python test.py
```

By default, the test script loads the model from `./runs/VTMF2N/` directory. Update the `model_path` in [test.py](./test.py) if needed.


## Model Zoo

The repository includes implementations of several models:

- VTMF2N (Our proposed model)
- CNN_LSTM
- CNN_MSTCN
- ViViT (Video Vision Transformer)
- TimeSformer
- C3D, R3D, R2Plus1D


## Citation

If you use this code in your research, please cite our paper.

## Acknowledgments

We thank the contributors and institutions that made this research possible. Special thanks to the datasets and open-source libraries used in this project.