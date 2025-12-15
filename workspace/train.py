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


loss_functions = {
    'MSELoss': nn.MSELoss,
    'L1Loss': nn.L1Loss,
    # 必要に応じて他の損失関数を追加
}


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


def build_model_and_optim(cfg):
    model = CAE(cfg)
    lr = float(cfg['train']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train_one_epoch(model, loader, loss_func, optimizer, device, epoch, writer, logger):
    model.train()
    total_loss = 0.0
    num_samples = len(loader.dataset)

    progress = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False, dynamic_ncols=True)
    for imgs in progress:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = loss_func(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    avg_loss = total_loss / num_samples
    writer.add_scalar('Loss/train', avg_loss, epoch)
    logger.info(f"Epoch {epoch} | train_loss={avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, loss_func, device, epoch, writer, logger, split='val', log_images=False, max_samples=10):
    model.eval()
    total_loss = 0.0
    num_samples = len(loader.dataset)
    first_in, first_out = None, None

    progress = tqdm(loader, desc=f"[{split.capitalize()}] Epoch {epoch}", leave=False, dynamic_ncols=True)
    for imgs in progress:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = loss_func(outputs, imgs)

        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        progress.set_postfix(loss=f"{batch_loss:.4f}")

        if first_in is None:
            first_in, first_out = imgs.detach().cpu(), outputs.detach().cpu()

    avg_loss = total_loss / num_samples
    writer.add_scalar(f'Loss/{split}', avg_loss, epoch)
    logger.info(f"Epoch {epoch} | {split}_loss={avg_loss:.6f}")

    # 画像ログ
    if log_images and first_in is not None and first_out is not None:
        import torchvision.utils as vutils
        n = min(max_samples, first_in.size(0))
        idx = torch.randperm(first_in.size(0))[:n]
        inp = first_in[idx]
        out = first_out[idx].clamp(0, 1)

        grid_in = vutils.make_grid(inp, nrow=min(n, 5), padding=2)
        grid_out = vutils.make_grid(out, nrow=min(n, 5), padding=2)
        writer.add_image(f'{split}/Input', grid_in, global_step=epoch)
        writer.add_image(f'{split}/Output', grid_out, global_step=epoch)
        logger.info(f"Logged sample images ({split}, epoch {epoch}, n={n})")

    return avg_loss


def main():
    # 設定ファイルの読み込み
    config_path = '/workspace/config/train.yaml'
    cfg = load_config(config_path)
    
    result_dir = Path(cfg['train']['result_dir']) / cfg['train']['experiment_name']
    # ロギング設定
    log_dir = result_dir / 'logs'
    logger, log_file = setup_logger(log_dir)
    
    logger.info("=" * 80)
    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config loaded from {config_path}")
    logger.info("=" * 80)

    # 事前検証
    try:
        validate_config(cfg, logger)
    except ValueError as e:
        logger.error(str(e))
        return

    # 設定ファイルのバックアップ
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, result_dir / 'config.yaml')


    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # データセット
    try:
        train_ds, val_ds, test_ds, channels = build_datasets(cfg)
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Val dataset size: {len(val_ds)}")
        if test_ds is not None:
            logger.info(f"Test dataset size: {len(test_ds)}")
    except Exception as e:
        logger.error(f"Failed to build datasets: {e}")
        return

    batch_size = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # モデル・最適化
    try:
        model, optimizer = build_model_and_optim(cfg)
        model = model.to(device)
        logger.info(f"Model initialized on {device}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return

    loss_func = loss_functions[cfg['train']['loss_function']]()

    # TensorBoard SummaryWriter
    tb_dir = result_dir / 'tb_summaries'
    writer = SummaryWriter(str(tb_dir))
    logger.info(f"TensorBoard logs saved to {tb_dir}")

    # ループ設定
    epochs = int(cfg['train']['epochs'])
    ckpt_dir = result_dir / 'ckpts'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    loss_table = []  # ロス履歴をテーブル形式で保存
    
    logger.info("=" * 80)
    logger.info(f"{'Epoch':<6} {'Train Loss':<15} {'Val Loss':<15} {'Status':<20}")
    logger.info("=" * 80)

    for epoch in range(1, epochs + 1):
        try:
            train_loss = train_one_epoch(model, train_loader, loss_func, optimizer, device, epoch, writer, logger)
            val_loss = evaluate(model, val_loader, loss_func, device, epoch, writer, logger,
                                split='val', log_images=True, max_samples=10)

            status = ""
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = ckpt_dir / f"cae_best.pth"
                torch.save(model.state_dict(), ckpt_path)
                status = "★ Best Updated"
                logger.info(f"Saved best checkpoint: {ckpt_path}")

            # ロス履歴に追加
            loss_table.append({
                'epoch': epoch,
                'train_loss': f"{train_loss:.6f}",
                'val_loss': f"{val_loss:.6f}",
                'status': status
            })

            # テーブル形式で表示
            logger.info(f"{epoch:<6} {train_loss:<15.6f} {val_loss:<15.6f} {status:<20}")

            # 定期保存
            latest_ckpt = ckpt_dir / f"cae_epoch_{epoch}.pth"
            torch.save(model.state_dict(), latest_ckpt)

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}")
            continue

    # 任意でテスト評価
    if test_ds is not None:
        try:
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loss = evaluate(model, test_loader, loss_func, device, epochs, writer, logger, split='test')
            logger.info(f"Test loss: {test_loss:.6f}")
        except Exception as e:
            logger.error(f"Failed to evaluate on test set: {e}")

    writer.close()

    # ロス履歴をCSVとして保存
    import csv
    loss_csv_path = log_dir / "loss_history.csv"
    try:
        with open(loss_csv_path, 'w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'status'])
            writer_csv.writeheader()
            writer_csv.writerows(loss_table)
        logger.info(f"Loss history saved to {loss_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save loss history: {e}")

    logger.info("=" * 80)
    logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()