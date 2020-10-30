import numpy as np
import torch
import torch.nn as nn
from FaultyMemory.perturbator import Perturbator
from abc import ABC, abstractclassmethod

REPR_DICT = {}
def add_repr(func):
    REPR_DICT[func.__name__] = func
    return func

class Representation(ABC):
    r""" Base class for custom representations
    """
    __COMPAT__ = None
    def __init__(self, width: int = 8):
        super().__init__()
        self.width = width

    def compatibility(self, other: Perturbator) -> bool:
        return self.__COMPAT__ in other.repr_compatibility

    @abstractclassmethod
    def encode(self, tensor: nn.Tensor) -> nn.Tensor:
        pass

    @abstractclassmethod
    def decode(self, tensor: nn.Tensor) -> nn.Tensor:
        pass

    def to_json(self):
        r"""
        Creates and returns a dictionnary with the necessary information to
        re-construct this instance
        """
        dict = {'name': type(self).__name__, 
                'width': self.width}
        return dict

@add_repr
class AnalogRepresentation(Representation):
    __COMPAT__ = 'ANALOG'
    def __init__(self) -> None:
        r''' A special case representation for e.g. memristors perturbations
        '''
        super.__init__(width=0.)

    def encode(self, tensor: nn.Tensor) -> nn.Tensor:
        return tensor

    def decode(self, tensor: nn.Tensor) -> nn.Tensor:
        return tensor

class DigitalRepresentation(Representation):
    __COMPAT__ = 'DIGITAL'
    pass

@add_repr
class BinaryRepresentation(DigitalRepresentation):
    def __init__(self) -> None:
        super().__init__(width=1)

    def encode(self, tensor: nn.Tensor) -> nn.Tensor:
        tensor = torch.sign(tensor) + 2 - 1
        return tensor.to(torch.uint8)

    def decode(self, tensor: nn.Tensor) -> nn.Tensor:
        return (tensor * 2) - 1

@add_repr
class ScaledBinaryRepresentation(DigitalRepresentation):
    def __init__(self) -> None:
        super().__init__(width=1)

    def encode(self, tensor: nn.Tensor) -> nn.Tensor:
        self.mean = torch.mean(torch.abs(tensor))
        tensor = torch.sign(tensor) + 2 - 1
        return tensor.to(torch.uint8)

    def decode(self, tensor: nn.Tensor) -> nn.Tensor:
        return ((tensor * 2) - 1) * self.mean

@add_repr
class ClusteredRepresentation(DigitalRepresentation):
    def __init__(self, num_cluster: int = 4) -> None:
        next = 2 ** np.ceil(np.log(num_cluster)/np.log(2))
        if next != num_cluster:
            Warning.warn('Number of cluster not fully using all bits, rounding up to the nearest power of 2 (max=8)')
        super().__init__(width=min(np.log(next)/np.log(2), 8))

    def encode(self, tensor: nn.Tensor) -> nn.Tensor:
        from scipy.cluster.vq import vq, kmeans, whiten
        whitened = whiten(tensor.clone().cpu().numpy().flatten())
        self.codebook, _ = kmeans(whitened, 2 ** self.width)
        cluster, _ = vq(whitened, self.codebook)
        return torch.tensor(cluster).view(tensor.shape).to(torch.uint8)

    def decode(self, tensor: nn.Tensor) -> nn.Tensor:
        shape = tensor.shape
        tensor = tensor.flatten()
        tensor = torch.gather(torch.from_numpy(self.codebook), 0, tensor)
        return tensor.view(shape)


def construct_repr(repr_dict, user_repr = {}):
    r"""
    Constructs a representation according to the dictionnary provided.
    The dictionnary should have a field for 'name' equals to the name of the class, the width and wether or not it is unsigned.
    """
    all_repr = dict(REPR_DICT, **user_repr)
    if repr_dict is None:
        return None
    return all_repr[repr_dict["name"]](width=repr_dict["width"])
