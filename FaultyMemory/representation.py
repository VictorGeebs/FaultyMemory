import math
import numpy as np
import torch
import torch.nn as nn
from numbers import Number
from FaultyMemory.perturbator import Perturbator
from abc import ABC, abstractclassmethod
from torch.utils.cpp_extension import load

Cpp_Repr = load(name="Cpp_Repr", sources=["FaultyMemory/cpp/representation.cpp"])

REPR_DICT = {}


def add_repr(func):
    REPR_DICT[func.__name__] = func
    return func


class Representation(ABC):
    r"""Base class for custom representations"""
    __COMPAT__ = None

    def __init__(self, width: int = 8):
        super().__init__()
        self.width = width
        assert (
            width > 0 and width <= 8 or "FreebieQuantization" in self.__class__.__name__
        ), "Support for precision up to 8 bits only"

    def compatibility(self, other: Perturbator) -> bool:
        if other.width > 1:
            # First case, both repr and pert on multibits = should be same
            width_check = self.width == other.width
        else:
            # Scd case, pert is scalar = should inflate to match repr width
            width_check = True
            if self.width > 1:
                other.width_correction = self.width
        return self.__COMPAT__ in other.repr_compatibility and width_check

    # @abstractclassmethod TODO
    # def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
    #     pass

    @abstractclassmethod
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        r""" By convention, do not return the same tensor as is"""
        pass

    @abstractclassmethod
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        r""" By convention, do not return the same tensor as is"""
        pass

    def save_attributes(self, tensor: torch.Tensor) -> None:
        self._target_device = tensor.device
        self._target_dtype = tensor.dtype
        self._target_shape = tensor.shape

    def load_attributes(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._target_device, self._target_dtype)

    def to_json(self):
        r"""
        Creates and returns a dictionnary with the necessary information to
        re-construct this instance
        """
        dict = {"name": type(self).__name__, "width": self.width}
        return dict


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
    pass


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
        self.mean = torch.mean(torch.abs(tensor))
        tensor = torch.sign(tensor) + 2 - 1
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.load_attributes(((tensor * 2) - 1) * self.mean)


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

    def adjust_fixed_point(self, mini: Number, maxi: Number) -> None:
        greatest = max(-mini, maxi)
        whole = max(math.ceil(math.log(2 * greatest, 2)), 0)
        self.nb_integer = min(whole, self.width)
        self.nb_digits = max(self.width - self.nb_integer, 0)
        if self.nb_digits == 0:
            print("Saturated range")

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
        self.save_attributes(tensor)
        return Cpp_Repr.encodeTenFixedPoint(tensor, self.width, self.nb_digits)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return Cpp_Repr.decodeTenFixedPoint(tensor, self.width, self.nb_digits)


@add_repr
class UFixedPointRepresentation(FixedPointRepresentation):
    @property
    def max_repr(self):
        return 2 ** self.nb_integer - self.resolution

    @property
    def min_repr(self):
        return 0


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
        tensor.apply_(
            lambda x: torch.bitwise_not(torch.abs(x)) + 1 if x < 0 else x
        )  # 2s compl
        return tensor.to(torch.uint8)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.load_attributes(self.min_repr + self.resolution * tensor)


# TODO not sure it works. Nice not to repeat same code though if it does.
@add_repr
class USlowFixedPointRepresentation(
    SlowFixedPointRepresentation, UFixedPointRepresentation
):
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        tensor = self.clamp_and_shift(tensor)
        return tensor.to(torch.uint8)


@add_repr
class ClusteredRepresentation(DigitalRepresentation):
    def __init__(self, num_cluster: int = 4) -> None:
        next = 2 ** np.ceil(np.log(num_cluster) / np.log(2))
        if next != num_cluster:
            Warning.warn(
                "Number of cluster not fully using all bits, rounding up to the nearest power of 2 (max=8)"
            )
        super().__init__(width=min(np.log(next) / np.log(2), 8))

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        self.save_attributes(tensor)
        # TODO pure pytorch impl of kmeans (avoid round trip to cpu)
        from scipy.cluster.vq import vq, kmeans, whiten

        whitened = whiten(tensor.clone().cpu().flatten().numpy())
        self.codebook, _ = kmeans(whitened, 2 ** self.width)
        cluster, _ = vq(whitened, self.codebook)
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


def construct_repr(repr_dict, user_repr={}):
    r"""
    Constructs a representation according to the dictionnary provided.
    The dictionnary should have a field for 'name' equals to the name of the class, the width of the repr
    """
    if repr_dict is None:
        return None
    all_repr = dict(REPR_DICT, **user_repr)
    return all_repr[repr_dict["name"]](width=repr_dict["width"])
