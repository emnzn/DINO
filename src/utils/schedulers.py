from typing import Dict, List

import torch
import numpy as np

from .network import Encoder

def get_param_groups(encoder: Encoder) -> Dict[str, List[torch.Tensor]]:
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


def get_teacher_temperatures(
    epochs: int, 
    num_warmup_epochs: int, 
    base_value: float, 
    final_value: float
    ) -> np.ndarray:
    """
    Constructs the list of temperatures for the teacher according to the 
    configured schedule.

    Parameters
    ----------
    epochs: int
        The total number of training epochs.

    num_warmup_epochs: int
        The number of warmup epochs.

    base_value: float
        The base temperature value.

    final_value: float
        The final temperature value.

    Returns
    -------
    temperatures: np.ndarray
        The array of temperatures per epoch.
    """

    if num_warmup_epochs > 0:
        assert base_value != final_value, "start and end must be differemt when num_warmup_epochs > 0"

    if num_warmup_epochs == 0:
        assert base_value == final_value, "Start and end must be the same when num_warmup_epochs == 0"

    warmup = np.linspace(start=base_value, stop=final_value, num=num_warmup_epochs)
    remaining = np.ones(epochs - num_warmup_epochs) * final_value

    temperatures = np.concatenate([warmup, remaining], axis=0)

    return temperatures