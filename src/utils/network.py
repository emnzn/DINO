import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import resnet50
from timm.models.vision_transformer import (
    vit_small_patch8_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_base_patch8_224
    )


class ProjectionHead(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

class Encoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 2048, 
        bottleneck_dim: int = 256,
        k_dim: int = 65536,
        num_layers: int = 3
        ):
        super().__init__()

        model = vit_small_patch16_224(dynamic_img_size=True)
        embded_dim = list(model.children())[-1].in_features

        mlp = ProjectionHead(
            nn.Linear(embded_dim, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.apply(self._init_weights)
        self.k_projection = weight_norm(nn.Linear(bottleneck_dim, k_dim, bias=False))
        self.k_projection.parametrizations.weight.original0.data.fill_(1)

        model.head_drop = nn.Identity()
        model.head = mlp

        self.encoder = model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.k_projection(x)

        return x