import torch
import matplotlib.pyplot as plt
import numpy as np
from models import CAE

def visualize_weights(model_path=None):
    # モデルのインスタンス化
    model = CAE()
    
    # 学習済みモデルがある場合はロードする
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("Using initialized weights (not trained)")

    model.eval()

    # パラメータ（重み）の取得と可視化
    # Conv2d層の重みのみを対象とする例
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]

    for i, layer in enumerate(conv_layers):
        weights = layer.weight.data.cpu().numpy()
        print(f"Layer {i+1} weight shape: {weights.shape}")
        
        # 最初の数個のフィルタを可視化
        num_filters = min(weights.shape[0], 16)
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
        fig.suptitle(f'Layer {i+1} Filters')
        
        for j in range(num_filters):
            # 最初の入力チャンネルの重みを表示
            filter_img = weights[j, 0, :, :]
            
            if num_filters == 1:
                ax = axes
            else:
                ax = axes[j]
                
            ax.imshow(filter_img, cmap='gray')
            ax.axis('off')
            
        plt.show()

if __name__ == "__main__":
    # 学習済みモデルのパスを指定する場合は引数に渡す
    # visualize_weights("path/to/model.pth")
    visualize_weights()