"""Basics utils to transfer a NN to another dataset. Please consider carefully the effects."""
import torch.nn as nn


def change_model_output(
    model: nn.Module, target_output_size: int, freeze_features=True
) -> nn.Module:
    """Replace latest ops of `model` output dimension with `target_output_size`."""
    if freeze_features:  # Needs to happen before the last layer is replaced
        freeze(model)
    ops = list(model.children())[-1]
    if isinstance(ops, nn.Linear):
        ops_prime = nn.Linear(
            ops.in_features, target_output_size, bias=(ops.bias is not None)
        )
        return nn.Sequential(*list(model.children())[:-1] + [ops_prime])
    else:
        raise ValueError(f"Last ops {type(ops)} is not supported yet")


def freeze(model: nn.Module):
    """Blindly set all parameters to requires_grad=False.

    Args:
        model (nn.Module): [description]
    """
    state = []
    for param in model.parameters():
        state.append(param.requires_grad)
        param.requires_grad = False
    model._unfreezed_state = state


def unfreeze(model: nn.Module):
    assert hasattr(model, '_unfreezed_state'), 'Cannot unfreeze what has not been'
    state = model._unfreezed_state
    for idx, param in enumerate(model.parameters()):
        param.requires_grad = state[idx]
