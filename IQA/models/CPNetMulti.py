'''
CPNet means ConvNet plus Patch-level head.
'''
import torch
import torch.nn as nn

import timm
from einops.layers.torch import Rearrange


class CPNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # load backbone
        self.backbone = timm.create_model('convnext_tiny', pretrained=args.pretrain, drop_path_rate=args.drop_path, features_only=True)
        self.rerange_layer = Rearrange('b c h w -> b (h w) c')
        self.avg_pool = nn.AdaptiveAvgPool2d(args.csize // 32)    # Max / Avg
        
        # Adaptive head
        embed_dim = 1440
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
            nn.ReLU()
        )
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # extract multi-scale feats & concat as hyper-column
        feats = self.backbone(x)
        feats = [self.avg_pool(feat) for feat in feats]
        feats = torch.cat(feats, dim=1) 
        feats = self.rerange_layer(feats)    # (b, c, h, w) -> (b, h*w, c)
        assert feats.shape[-1]==1440 and len(feats.shape)==3, 'Unexpected stacked features: {}'.format(feats.shape)

        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        y = torch.sum(scores*weights, dim=1) / torch.sum(weights, dim=1)

        return y


if __name__ == '__main__':
    # from utils import args
    from config import args

    x = torch.randn((2, 3, 224, 224))
    net = CPNet(args)
    y = net(x)

    print(y.shape)
