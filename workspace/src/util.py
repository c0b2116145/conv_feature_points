import yaml
import logging
from pathlib import Path
from datetime import datetime

import torch

from src.models import CAE

# ロギング設定
def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


# 追加: コンフィグ検証
def validate_config(cfg, logger):
    errors = []

    # dataset側の必須項目
    ds = cfg.get('dataset', {})
    ch = ds.get('channels', None)
    imsz = ds.get('image_size', None)
    if ch is None or imsz is None:
        errors.append("dataset.channels と dataset.image_size を設定してください。")

    # model側のチャネル整合性を事前チェック
    try:
        model_conf = cfg['model']
        enc_first_in = model_conf['encoder_layers'][0]['in_channels']
        final_out = model_conf['final_layer']['out_channels']
        if ch is not None and enc_first_in != ch:
            errors.append(f"encoderの最初のin_channels({enc_first_in})がdataset.channels({ch})と一致しません。")
        if ch is not None and final_out != ch:
            errors.append(f"final_layerのout_channels({final_out})がdataset.channels({ch})と一致しません。")
    except Exception as e:
        errors.append(f"model設定の読み込みに失敗: {e}")

    # 実際にモデルを通して入出力形状を検証
    if not errors and ch is not None and imsz is not None:
        try:
            model = CAE(cfg)
            model.eval()
            dummy = torch.zeros(1, ch, imsz, imsz)
            with torch.no_grad():
                out = model(dummy)
            # 1. チャネル一致
            if out.shape[1] != ch:
                errors.append(f"出力チャネルが入力と不一致: input={ch}, output={out.shape[1]}")
            # 2. 画像サイズ一致
            if (out.shape[2], out.shape[3]) != (imsz, imsz):
                errors.append(
                    f"出力画像サイズが入力と不一致: input={imsz}x{imsz}, output={out.shape[2]}x{out.shape[3]}"
                )
        except Exception as e:
            errors.append(f"モデル検証に失敗: {e}")

    if errors:
        error_msg = "Config validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Config validation passed.")


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)