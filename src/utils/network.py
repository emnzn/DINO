from typing import Dict, List

import torch
import numpy as np
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
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.k_projection(x)

        return x
    

class DINO(L.LightningModule):
    """
    A lightning implementation of DINO: self-distillation with no labels.

    Parameters
    ----------
    student: Encoder
        The initialized student encoder.

    teacher: Encoder
        The initialized teacher encoder.

    lr_schedule: np.ndarray
        An array consisting of the learning rate across each iteration step.
        Adjusted with Cosine Scheduling.

    teacher_temp_schedule: np.ndarray
        An array consisting of the temperatures used to sharpen the teacher outputs across epochs.
    
    weight_decay_schedule: np.ndarray
        An array consisting of the weight decay per iteration step.
        Adjusted with Cosine Scheduling.

    teacher_momentum_schedule: np.ndarray
        An array consisting of the momentum used to update the teacher per iteration step.
        Adjusted with Cosine Scheduling.

    param_groups: Dict[List[torch.Tensor]]
        The param groups for the optimizer.
        Decouples normalization and bias parameters from regularizable parameters.

    student_temp: float
        The temperature applied to the outputs of the student.

    center_momentum: float
        The momentum used for centering.

    k_dim: int
        The output dimension of the encoder.
    """

    def __init__(
        self,
        student: Encoder,
        teacher: Encoder,
        lr_schedule: np.ndarray,
        teacher_temp_schedule: np.ndarray,
        weight_decay_schedule: np.ndarray,
        teacher_momentum_schedule: np.ndarray,
        param_groups: Dict[str, List[torch.Tensor]],
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        k_dim: int = 65536
        ):
        super().__init__()

        self.student = student
        self.teacher = teacher

        self.lr_schedule = torch.tensor(lr_schedule, dtype=torch.float32)
        self.teacher_temp_schedule = torch.tensor(teacher_temp_schedule, dtype=torch.float32)
        self.weight_decay_schedule = torch.tensor(weight_decay_schedule, dtype=torch.float32)
        self.teacher_momentum_schedule = torch.tensor(teacher_momentum_schedule, dtype=torch.float32)

        self.param_groups = param_groups
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.center = torch.zeros(1, k_dim)

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def setup(self, stage=None):
        device = self.device

        self.lr_schedule = self.lr_schedule.to(device)
        self.teacher_temp_schedule = self.teacher_temp_schedule.to(device)
        self.weight_decay_schedule = self.weight_decay_schedule.to(device)
        self.teacher_momentum_schedule = self.teacher_momentum_schedule.to(device)

        self.student_temp = torch.tensor(self.student_temp).to(device)
        self.center_momentum = torch.tensor(self.center_momentum).to(device)
        self.center = self.center.to(device)

    def training_step(self, batch, _) -> torch.Tensor:
        current_epoch = self.current_epoch
        iteration = self.global_step
        optimizer = self.optimizers()

        teacher_temp = self.teacher_temp_schedule[current_epoch]
        views = batch["image"]
        global_crops = views["global_crops"]
        local_crops = views["local_crops"]
        
        global_crop0 = global_crops[0]
        global_crop1 = global_crops[1]

        with torch.inference_mode():
            teacher_outs = {
                "g0": self.teacher(global_crop0),
                "g1": self.teacher(global_crop1)
            }

        student_outs = {
            "g0": self.student(global_crop0),
            "g1": self.student(global_crop1)
        }
        for i, crop in enumerate(local_crops):
            student_out = self.student(crop)
            student_outs[f"l{i}"] = student_out

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[iteration]
            if i == 0:
                param_group["weight_decay"] = self.weight_decay_schedule[iteration]

        gathered_teacher_outs = self.gathered_teacher_outs(teacher_outs)

        self.center = self.update_center(
            teacher_out=gathered_teacher_outs,
            center=self.center,
            momentum=self.center_momentum
        )

        total_loss = 0
        n_loss_terms = 0

        for global_crop, teacher_out in teacher_outs.items():
            teacher_out = F.softmax(((teacher_out - self.center) / teacher_temp), dim=-1)

            for crop, student_out in student_outs.items():
                if global_crop != crop:
                    student_out = F.softmax((student_out / self.student_temp), dim=-1)

                    loss = -torch.sum(teacher_out * torch.log(student_out), dim=-1)
                    avg_loss = loss.mean()
                    total_loss += avg_loss
                    n_loss_terms += 1
        
        iteration_loss = total_loss / n_loss_terms

        # per step logging
        self.log("Learning Rate", self.lr_schedule[iteration], on_step=True, on_epoch=False, prog_bar=True)
        self.log("Weight Decay", self.weight_decay_schedule[iteration], on_step=True, on_epoch=False, prog_bar=True)
        self.log("Teacher Momentum", self.teacher_momentum_schedule[iteration], on_step=True, on_epoch=False, prog_bar=True)

        # per epoch logging
        self.log("Teacher Temperature", teacher_temp, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("Loss", iteration_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return iteration_loss

    def on_train_batch_end(self, *_):
        teacher_momentum = self.teacher_momentum_schedule[self.global_step - 1]
        self.update_teacher(self.teacher, self.student, teacher_momentum)
        
        if torch.cuda.device_count() > 1:
            self.synchronize_teacher()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.param_groups)

        return optimizer

    @torch.no_grad()
    def update_center(
        self,
        teacher_out: torch.Tensor,
        center: torch.Tensor,
        momentum: float
        ) -> torch.Tensor:
        
        center = momentum * center + (1 - momentum) * torch.mean(teacher_out, dim=0, keepdim=True)
        
        return center
    
    @torch.no_grad()
    def gathered_teacher_outs(
        self, 
        teacher_outs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        if self.trainer.world_size > 1:
            gathered_teacher_g0 = self.all_gather(teacher_outs["g0"])
            gathered_teacher_g1 = self.all_gather(teacher_outs["g1"])

            gathered_teacher_g0 = gathered_teacher_g0.reshape(-1, gathered_teacher_g0.shape[-1])
            gathered_teacher_g1 = gathered_teacher_g1.reshape(-1, gathered_teacher_g1.shape[-1])

            gathered_teacher_outs = torch.cat([gathered_teacher_g0, gathered_teacher_g1], dim=0)
    
        else:
            gathered_teacher_outs = torch.cat([v for _, v in teacher_outs.items()], dim=0)

        return gathered_teacher_outs
    
    @torch.no_grad()
    def update_teacher(
        self, 
        teacher: Encoder, 
        student: Encoder, 
        teacher_momentum: float
        ):

        """
        Updates the teacher using a moving average of the student's parameters.

        Parameters
        ----------
        teacher: Encoder

        student: Encoder

        momentum: float
            The momentum value between 0 - 1 where a higher value results in a 
            lower sensitivity to new updates.
        """

        for param_t, param_s in zip(teacher.parameters(), student.parameters()):
            param_t.data.mul_(teacher_momentum).add_((1 - teacher_momentum) * param_s.data)

    @torch.no_grad()
    def synchronize_teacher(self):
        for param in self.teacher.parameters():
            torch.distributed.broadcast(param.data, src=0)



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

def get_encoders(backbone: str) -> Encoder:
    """
    Returns an initialized student and teacher network.
    """
    
    student = Encoder(backbone)
    
    teacher = Encoder(backbone)
    for param in teacher.parameters():
        param.requires_grad = False

    return student, teacher