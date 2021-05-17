"""Basics utils to transfer a NN to another dataset. Please consider carefully the effects."""
from typing import Callable, Optional, Union
import torch
import torch.nn as nn


def change_model_output(model: nn.Module, target_output_size: int, freeze_features=True) -> nn.Module:
    """Replace latest ops of `model` output dimension with `target_output_size`.
     """
    if freeze_features:
        freeze(model)
    ops = list(model.children())[-1]
    if isinstance(ops, nn.Linear):
        ops_prime = nn.Linear(ops.in_features, target_output_size, bias=(ops.bias is not None))
        return nn.Sequential(*list(model.children())[:-1] + [ops_prime])
    else:
        raise ValueError('Last ops is not supported yet')


def freeze(model: nn.Module):
    """Blindly set all parameters to requires_grad=False.

    Args:
        model (nn.Module): [description]
    """
    for param in model.parameters():
        param.requires_grad = False
