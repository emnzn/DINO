import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.nn.utils.parametrizations import weight_norm
from timm.models.vision_transformer import (
    VisionTransformer,
    vit_small_patch8_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_base_patch8_224
    )


class Encoder(nn.Module):
    """
    Initializes the encoder with a configurable projection head.

    Parameters
    ----------
    backbone: str
        Must be one of [`resnet50`, `vit-s-8`, `vit-s-16`, `vit-b-8`, `vit-b-16`]

    hidden_dim: int
        The hidden dimension of the MLP.

    bottleneck_dim: int
        The bottleneck dimension.

    k_dim: int
        The dimension of the weight normalized final linear layer.

    Returns
    -------
    x: torch.Tensor
        The embedding of the samples passed to the encoder.
    """

    def __init__(
        self, 
        backbone: str,
        mlp_layers: int = 3,
        hidden_dim: int = 2048, 
        bottleneck_dim: int = 256,
        k_dim: int = 65536
        ):
        super().__init__()

        backbone_dim_table = {
            "resnet50": 2048,
            "vit-s-8": 384,
            "vit-s-16": 384,
            "vit-b-8": 768,
            "vit-b-16": 768
        }

        self.encoder = get_model(backbone)
        embed_dim = backbone_dim_table[backbone]

        self.apply(self._init_weights)
        
        self.mlp = ProjectionHead(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(mlp_layers - 2)],
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        self.k_projection = weight_norm(nn.Linear(bottleneck_dim, k_dim, bias=False))
        self.k_projection.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.mlp(x)
        x = self.k_projection(x)

        return x
    

class DINO(L.LightningModule):

    def __init__(
        self,
        encoder: Encoder,
        learning_rate: float,
        weight_decay: float,
        eta_min: float,
        temperature: float
        ):
        super().__init__()

        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.temperature = temperature

    def training_step(self, batch, _):
        pass

    def configure_optimizers():
        pass


class ResNet50(nn.Module):
    """
    A constructor for the ResNet Encoder.
    """

    def __init__(self):
        super().__init__()

        model = resnet50()
        backbone = list(model.children())[:-1]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        return x


class ProjectionHead(nn.Sequential):
    """
    Just used for better annotations in the model graph visualization.
    """

    def __init__(self, *args):
        super().__init__(*args)


def get_model(backbone: str) -> ResNet50 | VisionTransformer:
    """
    Function to return a model to be passed as a component to the Encoder.

    Parameters
    ----------
    backbone: str
        Must be one of [`resnet50`, `vit-s-8`, `vit-s-16`, `vit-b-8`, `vit-b-16`].

    Returns
    -------
    model: ResNet50 | VisionTransformer
        The initialized model.
    """

    model_table = {
        "resnet50": ResNet50(),
        "vit-s-8": vit_small_patch8_224(dynamic_img_size=True),
        "vit-s-16": vit_small_patch16_224(dynamic_img_size=True),
        "vit-b-8": vit_base_patch16_224(dynamic_img_size=True),
        "vit-b-16": vit_base_patch8_224(dynamic_img_size=True)
    }

    assert backbone in model_table, f"backbone must be one of {list(model_table.keys())}"

    model = model_table[backbone]

    if backbone != "resnet50":
        model.fc_norm = nn.Identity()
        model.head_drop = nn.Identity()
        model.head = nn.Identity()

    return model