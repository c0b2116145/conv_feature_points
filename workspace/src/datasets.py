import json
from pathlib import Path
from typing import Callable, List, Union, Dict, Any, Optional

from PIL import Image
from torch.utils.data import Dataset


def _extract_paths(data: Union[List[Any], Dict[str, Any]]) -> List[str]:
    """JSONの形式に柔軟に対応して画像パスのリストを取り出す"""
    if isinstance(data, list):
        # ["a.jpg", ...] もしくは [{"file_name": "a.jpg"}, ...]
        if len(data) > 0 and isinstance(data[0], dict) and "file_name" in data[0]:
            return [d["file_name"] for d in data]
        return data
    if isinstance(data, dict):
        images = data.get("images", [])
        if isinstance(images, list):
            if len(images) > 0 and isinstance(images[0], dict) and "file_name" in images[0]:
                return [d["file_name"] for d in images]
            return images
    raise ValueError("Unsupported JSON structure for image list")


class PascalImageDataset(Dataset):
    """
    JSON で列挙された画像を読み込む Dataset（ラベル無し、AE/CAE 用）
    - json_path: 画像リストが書かれた JSON ファイルへのパス
    - root_dir: 画像の実体があるルートパス（相対パス解決用）
    - transform: 画像に適用する torchvision.transforms 等
    """
    def __init__(
        self,
        json_path: Union[str, Path],
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.json_path = Path(json_path)
        self.root_dir = Path(root_dir)
        self.transform = transform

        with self.json_path.open("r", encoding='utf_8_sig') as f:
            data = json.load(f)
        self.image_relpaths = _extract_paths(data)
        if len(self.image_relpaths) == 0:
            raise ValueError(f"No images found in {self.json_path}")

    def __len__(self):
        return len(self.image_relpaths)

    def __getitem__(self, idx: int):
        rel = self.image_relpaths[idx]
        img_path = self.root_dir / rel
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 画像を返す
        return img