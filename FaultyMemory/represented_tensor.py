from FaultyMemory.utils import dictify, ten_exists
from FaultyMemory.perturbator import Perturbator
from typing import Dict, Optional, Tuple, Union
from FaultyMemory.representation import Representation
import numpy as np
import torch.nn as nn
from abc import ABC

class RepresentedTensor(ABC):
    def __init__(self, model: nn.Module, name: str, repr: Representation = None, pert: Optional[Union[Dict, Perturbator]] = {}) -> None:
        self.name = name
        self.repr = repr
        self.pert = dictify(pert)
        self.model = model
        ten_exists(self.where_ten(), name)
        self.compute_bitcount()

    def access_ten(self):
        return self.where_ten()[self.name]

    def to_repr(self, x) -> None:
        encoded = self.repr.encode(x)
        perturbed = self.pert(encoded)
        return self.repr.decode(perturbed).to(x.dtype)

    def apply_perturb_to_encoded(self, base) -> nn.Tensor:
        for pert in self.pert:
            if not pert:
                continue
            assert self.repr.compatibility(pert), 'The perturbation is not compatible with the representation'
            base = pert(base, self.repr.width)
        return base

    def quantize_perturb(self) -> None:
        r''' Overloaded operator that manages the quantization
        '''
        pass

    def restore(self) -> None:
        r''' Overloaded operator that manages the restoration
        '''
        #TODO is useful ? can be managed from outside (otherwise RepresentedParameter needs to hold save)
        pass

    def where_ten(self) -> dict: 
        pass

    def compute_bitcount(self) -> None:
        r''' Overloaded operator that set self.bitcount
        '''
        pass

    def energy_consumption(self, a=12.8) -> Tuple(int, float):
        assert self.bitcount is not None, 'Bitcount has not been set in `compute_bitcount`'
        if 'BitwisePert' in self.pert:
            p = self.pert['BitwisePert'].p
        else:
            print('There are no consumption model other than for Bitwise pert yet')
            p = 0.
        current_consumption = -np.log(p/a) if p > 0 else 1
        return self.bitcount, self.bitcount * current_consumption


class RepresentedParameter(RepresentedTensor):
    r""" Seamlessly cast a parameter tensor to faulty hardware
    """
    def compute_bitcount(self) -> None:
        self.bitcount = self.access_ten().numel() * self.repr.width

    def where_ten(self) -> dict:
        return self.model.named_parameters()

    def quantize_perturb(self) -> None:
        ten = self.access_ten()
        ten_prime = self.to_repr(ten)
        ten.data.copy_(ten_prime.data)


class RepresentedActivation(RepresentedTensor):
    r""" Seamlessly cast an activation tensor to faulty hardware
    """
    def compute_bitcount(self) -> None:
        def hook_bitcount(self, module, input, output):
            self.bitcount = (output.numel() / output.shape[0]) * self.repr.width
            hook.remove()
        hook = self.access_ten().register_forward_hook(hook_bitcount)

    def where_ten(self) -> dict:
        return self.model.named_modules()

    def quantize_perturb(self) -> None:
        if not self.hook:
            def hook(self, module, input, output):
                output.data.copy_(self.to_repr(output).data)
            self.hook = self.access_ten().register_forward_hook(hook)

    def __del__(self):
        if self.hook:
            self.hook.remove()
        super().__del__()


class RepresentedModule_(RepresentedTensor):
    r''' Replace a module with a represented one
    TODO the goal is to not be seamless, i.e. the network definition changes
    '''
    raise NotImplementedError('In place module replacement is not supported yet')