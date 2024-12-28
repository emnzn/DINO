from typing import Dict, List

import torch
import numpy as np

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


def cosine_scheduling(
    base_value: float, 
    final_value: float, 
    epochs: int, 
    iters_per_epoch: int,
    warmup_epochs: int = 0, 
    start_warmup_value: float = 0.0
    ) -> np.ndarray:
    """
    A function to return the value of a parameter across timesteps 
    when adjusted with cosine annealing scheduler.

    Parameters
    ----------
    base_value: float
        The starting value of the parameter.

    final_value: float
        The final value of the parameter.

    epochs: int
        The number of total epochs.
    
    iters_per_epoch: int
        The number of steps per epoch.

    warmup_epochs: int
        The number of epochs to perform a linear warmup.

    start_warmup_value: float
        The starting value of the parameter when perfoming initial warmup.

    Returns
    -------
    schedule: np.ndarray
        An array containing the parameter value across epochs.
    """

    warm_up = np.array([])
    warm_up_iters = warmup_epochs * iters_per_epoch
    
    if warmup_epochs > 0:
        warm_up = np.linspace(start=start_warmup_value, stop=base_value, num=warm_up_iters)

    iters = np.arange(epochs * iters_per_epoch - warm_up_iters)
    cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos((iters * np.pi) / len(iters)))

    schedule = np.concatenate((warm_up, cosine))

    return schedule
