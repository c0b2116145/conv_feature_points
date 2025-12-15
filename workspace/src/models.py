import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, config):
        super(CAE, self).__init__()
        
        model_conf = config['model']

        # --- Encoder ---
        # 設定リストに基づいて層を追加
        encoder_layers = []
        for layer_cfg in model_conf['encoder_layers']:
            encoder_layers.append(nn.Conv2d(
                layer_cfg['in_channels'], 
                layer_cfg['out_channels'], 
                layer_cfg['kernel_size'], 
                padding=layer_cfg['padding']
            ))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool2d(kernel_size=layer_cfg['Maxpool_kernel_size']))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Decoder ---
        decoder_layers = []
        for layer_cfg in model_conf['decoder_layers']:
            decoder_layers.append(nn.Conv2d(
                layer_cfg['in_channels'], 
                layer_cfg['out_channels'], 
                layer_cfg['kernel_size'], 
                padding=layer_cfg['padding']
            ))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Upsample(scale_factor=layer_cfg['Upsample_scale_factor']))
            
        # 最終層 (Upsampleなし、Sigmoidあり)
        final_cfg = model_conf['final_layer']
        decoder_layers.append(nn.Conv2d(
            final_cfg['in_channels'],
            final_cfg['out_channels'],
            final_cfg['kernel_size'],
            padding=final_cfg['padding']
        ))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x