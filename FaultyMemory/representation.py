r""" Classes to represent numbers in memory.

Each usable class is stored in `REPR_DICT`
Can be Digital or Analog representation
They at least expose two methods: encode and decode

Encode returns a tensor in uint8 format
Decode takes a tensor in uint8 format
"""
from FaultyMemory.utils import twos_compl
import math
import numpy as np
import torch
import logging
from .perturbator import Perturbator
from abc import ABC, abstractclassmethod

# from torch.utils.cpp_extension import load
from typing import Callable

# Cpp_Repr = load(name="Cpp_Repr", sources=["FaultyMemory/cpp/representation.cpp"])

REPR_DICT = {}


def add_repr(func: Callable) -> Callable:
    r"""Decorator to populate `REPR_DICT`

    Args:
        func (Callable): The representation class

    Returns:
        Callable: The same representation class
    """
    REPR_DICT[func.__name__] = func
    return func


class Representation(ABC):
    r"""Base class for custom representations"""
    __COMPAT__ = "None"

    def __init__(self, width: int = 8):
        super().__init__()
        self.width = width
        assert (
            0 < width <= 8 or "FreebieQuantization" in self.__class__.__name__
        ), "Support for precision up to 8 bits only"

    def compatibility(self, other: Perturbator) -> bool:
        if other.width > 1:
            # First case, both repr and pert on multibits = should be same
            width_check = self.width == other.width
        elif other.width == 1:
            # Scd case, pert is scalar = should inflate to match repr width
            width_check = True
            if self.width > 1:
                other.width_correction = self.width
        else:
            raise ValueError("Perturbator width should be an integer greather than 0")
        return self.__COMPAT__ in other.repr_compatibility and width_check

    @abstractclassmethod
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        r""" By convention, do not return the same tensor as is"""

    @abstractclassmethod
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        r""" By convention, do not return the same tensor as is"""

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.decode(self.encode(tensor))

    def save_attributes(self, tensor: torch.Tensor) -> None:
        self._target_device = tensor.device
        self._target_dtype = tensor.dtype
        self._target_shape = tensor.shape

    def load_attributes(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._target_device, self._target_dtype)

    def to_json(self) -> dict:
        r"""
        Creates and returns a dictionnary with the necessary information to
        re-construct this instance
        """
        _dict = {"name": type(self).__name__, "width": self.width}
        return _dict


class JustQuantize(Representation):
    r"""Subclass and define encode to an arbitrary representation
    Will not work with any repr ! (since we `JustQuantize`)
    """

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


@add_repr
class FreebieQuantization(JustQuantize):
    def __init__(self) -> None:
        r"""A special case representation for non quantized tensors still stored in memories. Consider 16 bit per item, supposed not to degrade the performance."""
        super().__init__(width=16)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


@add_repr
class AnalogRepresentation(Representation):
    __COMPAT__ = "ANALOG"

    def __init__(self) -> None:
        r"""A special case representation for e.g. memristors perturbations"""
        super().__init__(
            width=1
        )  # width is 1 in this case, each value of the tensor is a 1d float

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


class DigitalRepresentation(Representation):
    __COMPAT__ = "DIGITAL"


@add_repr
class BinaryRepresentation(DigitalRepresentation):
    def __init__(self, **kwargs) -> None:
        super().__init__(width=1)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        tensor = (torch.sign(tensor) + 1) / 2
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.load_attributes((tensor.to(torch.float32) * 2) - 1)


@add_repr
class ScaledBinaryRepresentation(DigitalRepresentation):
    def __init__(self) -> None:
        super().__init__(width=1)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        self.mean = torch.mean(torch.abs(tensor)).item()
        tensor = (torch.sign(tensor) + 1) / 2
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.load_attributes(((tensor.to(torch.float32) * 2) - 1) * self.mean)


@add_repr
class FixedPointRepresentation(DigitalRepresentation):
    r"""Signed Fixed Point impl.
    By default, range is optimized in the interval +- 3.0
    """
    # FIXME Fail on HPC system with CUDA GPU
    def __init__(self, width=8, nb_digits=-1) -> None:
        super().__init__(width)
        if nb_digits >= 0:
            assert nb_digits <= width
            self.nb_digits = nb_digits
            self.nb_integer = width - nb_digits
        else:
            self.adjust_fixed_point(mini=-3, maxi=3)

    def adjust_fixed_point(self, mini: float, maxi: float) -> None:
        assert maxi > mini, "Maxi should be sup. to mini"
        greatest = max(math.floor(abs(mini)), math.floor(abs(maxi)))
        whole = max(math.ceil(math.log(2 * greatest, 2)), 0)
        self.nb_integer = min(whole, self.width)
        self.nb_digits = max(self.width - self.nb_integer, 0)
        if self.nb_digits == 0:
            logging.info("Saturated range")

    @property
    def resolution(self):
        return 2 ** (-self.nb_digits)

    @property
    def max_repr(self):
        return 2 ** (self.nb_integer - 1) - self.resolution

    @property
    def min_repr(self):
        return -(2 ** (self.nb_integer - 1))

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        pass
        # self.save_attributes(tensor)
        # return Cpp_Repr.encodeTenFixedPoint(tensor, self.width, self.nb_digits)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        pass
        # return Cpp_Repr.decodeTenFixedPoint(tensor, self.width, self.nb_digits)


@add_repr
class SlowFixedPointRepresentation(FixedPointRepresentation):
    "Pure Pytorch Python API FP Representation"

    def clamp_and_shift(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clamp(min=self.min_repr, max=self.max_repr)
        return torch.round_(tensor << self.nb_digits).to(
            torch.int16
        )  # to not lose the sign yet, and still apply bitwise func

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        tensor = self.clamp_and_shift(tensor)
        tensor = torch.where(tensor < 0, twos_compl(tensor), tensor)  # 2s compl
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.load_attributes(tensor)
        tensor_dec = tensor * self.resolution
        tensor = torch.where(
            tensor >= 2 ** (self.width - 1),
            tensor_dec - 2 ** (self.nb_integer),
            tensor_dec,
        )
        return tensor


@add_repr
class UFixedPointRepresentation(SlowFixedPointRepresentation):
    @property
    def max_repr(self):
        return 2 ** self.nb_integer - self.resolution

    @property
    def min_repr(self):
        return 0

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        tensor = self.clamp_and_shift(tensor)
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.load_attributes(tensor)
        return tensor * self.resolution


@add_repr
class ClusteredRepresentation(DigitalRepresentation):
    def __init__(self, num_cluster: int = 4) -> None:
        next = 2 ** np.ceil(np.log(num_cluster) / np.log(2))
        if next != num_cluster:
            logging.warning(
                "Number of cluster not fully using all bits, rounding up to the nearest power of 2 (max=8)"
            )
        super().__init__(width=min(np.log(next) / np.log(2), 8))

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        # TODO pure pytorch impl of kmeans (avoid round trip to cpu)
        from scipy.cluster.vq import vq, kmeans

        tensor = tensor.clone().cpu().flatten().numpy()
        self.codebook, _ = kmeans(tensor, 2 ** self.width)
        cluster, _ = vq(tensor, self.codebook)
        return (
            torch.tensor(cluster)
            .view(self._target_shape)
            .to(self._target_device, torch.uint8)
        )

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.flatten()
        tensor = torch.gather(
            torch.from_numpy(self.codebook), 0, tensor.to(torch.int64)
        )
        return self.load_attributes(tensor.view(self._target_shape))


def construct_repr(repr_dict: dict = None, user_repr: dict = None) -> Representation:
    r"""Construct a representation according to the dictionnary provided.

    The dictionnary should have a field for 'name' equals to the name of the class, the width of the repr
    If `repr_dict` is None, return a FreebieQuantization
    """
    if repr_dict is None:
        return FreebieQuantization()
    if user_repr is not None:
        all_repr = dict(REPR_DICT, **user_repr)
    else:
        all_repr = REPR_DICT
    return all_repr[repr_dict["name"]](width=repr_dict["width"])
