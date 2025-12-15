import yaml
import os
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

from src.models import CAE
from src.datasets import PascalImageDataset
from src.util import setup_logger, validate_config, load_config

def build_transforms(image_size: int, channels: int):
    # channels==1ならグレースケール化
    t = [
        transforms.Resize((image_size, image_size)),
    ]
    if channels == 1:
        t.append(transforms.Grayscale(num_output_channels=1))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def build_datasets(cfg):
    ds_cfg = cfg['dataset']
    image_size = int(ds_cfg['image_size'])
    channels = int(ds_cfg['channels'])

    transform = build_transforms(image_size, channels)

    # train
    train_json = Path(ds_cfg['train_json_path'])
    train_root = train_json.parent / ds_cfg['imagedir']
    train_ds = PascalImageDataset(train_json, train_root, transform)

    # val
    val_json = Path(ds_cfg['val_json_path'])
    val_root = val_json.parent / ds_cfg['imagedir']
    val_ds = PascalImageDataset(val_json, val_root, transform)

    # test（必要なら最後に評価）
    test_json = Path(ds_cfg['test_json_path'])
    test_root = test_json.parent / ds_cfg['imagedir']
    test_ds = None
    if test_json.exists():
        test_ds = PascalImageDataset(str(test_json), test_root, transform)

    return train_ds, val_ds, test_ds, channels


def main():
    config_path = '/workspace/config/train.yaml'
    cfg = load_config(config_path)
    train_ds, val_ds, test_ds, channels = build_datasets(cfg)
    batch_size = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    batch_data = next(iter(train_loader))
    print(f"Loaded batch data shape: {batch_data.shape}")
    print(f"Loaded batch data type: {batch_data.dtype}")
    image = batch_data[0]  # バッチの最初の画像を取得
    print(f"First image shape: {image.shape}, max: {image.max()}, min: {image.min()}")

if __name__ == "__main__":
    main()