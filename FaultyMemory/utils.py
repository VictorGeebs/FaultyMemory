import torch
import copy
import re
import math
from numbers import Number
import numpy as np


def listify(obj: object) -> list:
    if isinstance(obj, list):
        return obj
    elif obj is None:
        return []
    else:
        return [obj]


def dictify(obj: object) -> dict:
    if isinstance(obj, dict):
        return obj
    elif obj is None:
        return {}
    elif isinstance(obj, list):
        return {str(type(o).__name__): o for o in obj}
    else:
        return {str(type(obj).__name__): obj}


def typestr_to_type(obj):
    res = re.match(r"\.([A-z]*)'", str(type(obj)))
    if res:
        return res.group(1)
    else:
        res = re.match(r"'([A-z]*)'", str(type(obj)))
        return res.group(1)


def ten_exists(model_dict: dict, name: str) -> None:
    try:
        assert (
            name in model_dict
        ), f"Specified name {name} not in model_dict and no reference given, cannot add this tensor"
    except AssertionError as id:
        print(id)
        return


def sanctify_ten(ten: torch.Tensor) -> torch.Tensor:
    r"""Take any tensor, make a deepcopy of it, and ensure the deepcopy is on CPU

    Note: `deepcopy` is needed in case the tensor is already on CPU, in which case `.cpu()` do not make a copy
    """
    return copy.deepcopy(ten.clone().detach()).cpu()


def sanitize_number(
    value: Number,
    mini: Number = float("-inf"),
    maxi: Number = math.inf,
    rnd: bool = False,
) -> Number:
    if rnd:
        value = round(value)
    return mini if value < mini else maxi if value > maxi else value


def kmeans_nparray(np_array: np.array, nb_clusters: int) -> np.array:
    from scipy.cluster.vq import kmeans, vq, whiten

    whitened = whiten(np_array)
    codebook, _ = kmeans(whitened, nb_clusters)
    encoded, _ = vq(np_array, codebook)
    return np.array([codebook[i] for i in encoded])


@torch.jit.script
def twos_compl(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_not(tensor.abs()) + 1
