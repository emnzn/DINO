from typing import Dict, List

import torch

from .network import Encoder

def get_param_groups(encoder: Encoder) -> Dict[List[torch.Tensor]]:
    """
    Creates parameter groups to decouple weight decay.

    Parameters
    ----------
    encoder: Encoder
        An initialized Encoder to be optimized.

    Returns
    -------
    param_groups: Dict[List[torch.Tensor]]
        The parameter groups where the first group are regularizable,
        and the second are bias and normalization terms.

    Applied from: https://github.com/facebookresearch/dino/blob/main/utils.py
    """

    regularized = []
    not_regularized = []

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)

            else:
                regularized.append(param)

    param_groups = [
        {"params": regularized},
        {"params": not_regularized, "weight_decay": 0.0}
    ]

    return param_groups