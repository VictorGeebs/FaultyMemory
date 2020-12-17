from FaultyMemory.utils import dictify, ten_exists, sanctify_ten
from FaultyMemory.perturbator import Perturbator, construct_pert
from typing import Dict, Optional, Tuple, Union
from FaultyMemory.representation import Representation, construct_repr
import numpy as np
import torch.nn as nn
from abc import ABC, abstractclassmethod

TYPE_DICT = {}
def add_type(func):
    TYPE_DICT[func.__name__] = func
    return func

class RepresentedTensor(ABC):
    def __init__(self, model: nn.Module, name: str, repr: Representation, pert: Optional[Union[Dict, Perturbator]] = {}) -> None:
        self.name = name
        self.repr = repr
        self.pert = dictify(pert)
        self.model = model
        ten_exists(self.where_ten(), name)
        self.compute_bitcount()

    @classmethod
    def from_dict(cls, dict: dict, model: nn.Module):
        return cls(model, 
                   dict['name'],
                   construct_repr(dict['repr']),
                   {pert['name']: construct_pert(pert) for pert in dict['pert']})

    def access_ten(self):
        return self.where_ten()[self.name]

    def to_repr(self, x) -> None:
        # TODO mode d'opération sans les perturb ?
        encoded = self.repr.encode(x)
        assert encoded.shape == x.shape, 'The encoded version is not of the same shape as the input tensor'
        perturbed = self.apply_perturb_to_encoded(encoded)
        return self.repr.decode(perturbed).to(x.dtype)

    def apply_perturb_to_encoded(self, base) -> nn.Tensor:
        # TODO : si pert == 0, just quantize
        for pert in self.pert:
            if not pert:
                continue
            assert self.repr.compatibility(pert), 'The perturbation is not compatible with the representation'
            base = pert(base)
        return base

    @abstractclassmethod
    def quantize_perturb(self) -> None:
        r''' Overloaded operator that manages the quantization
        '''
        pass

    def save(self, ten) -> None:
        if self.saved_ten is None:
            self.saved_ten = sanctify_ten(ten)
            self.ref_ten = ten
        else:
            print("Another tensor is already saved") #TODO test if 2 GPU trigger this print

    def restore(self) -> None:
        if self.saved_ten is not None:
            self.ref_ten.data.copy_(self.saved_ten.data.to(self.ref_ten))
            del self.saved_ten

    @abstractclassmethod
    def where_ten(self) -> dict: 
        pass

    @abstractclassmethod
    def compute_bitcount(self) -> None:
        r''' Overloaded operator that set self.bitcount
        '''
        pass

    def energy_consumption(self, a=12.8) -> Tuple(int, float):
        assert self.bitcount is not None, 'Bitcount has not been set in `compute_bitcount`'
        if 'BitwisePert' in self.pert:
            p = self.pert['BitwisePert'].distribution.probs
        else:
            print('There are no consumption model other than for Bitwise pert yet')
            p = 0.
        current_consumption = -np.log(p/a) if p > 0 else 1.
        return self.bitcount, self.bitcount * current_consumption

    def to_json(self):
        return {'type': type(self).__name__,
                'name': self.name,
                'repr': self.repr.to_json(),
                'pert': [pert.to_json() for pert in self.pert]}


@add_type
class RepresentedParameter(RepresentedTensor):
    r""" Seamlessly cast a parameter tensor to faulty hardware
    """
    def compute_bitcount(self) -> None:
        self.bitcount = self.access_ten().numel() * self.repr.width

    def where_ten(self) -> dict:
        return self.model.named_parameters()

    def quantize_perturb(self) -> None:
        ten = self.access_ten()
        self.save(ten)
        ten_prime = self.to_repr(ten)
        ten.data.copy_(ten_prime.data)


@add_type
class RepresentedActivation(RepresentedTensor):
    r""" Seamlessly cast an activation tensor to faulty hardware
    TODO do not support save/restore yet
    """
    def compute_bitcount(self) -> None:
        def hook_bitcount(self, module, input, output):
            self.bitcount = (output.numel() / output.shape[0]) * self.repr.width
            self.hook_bitcount.remove()
            del self.hook_bitcount
        self.hook_bitcount = self.access_ten().register_forward_hook(hook_bitcount)

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
        if self.hook_bitcount:
            self.hook_bitcount.remove()
        super().__del__()


class RepresentedModule_(RepresentedTensor):
    r''' Replace a module with a represented one
    TODO the goal is to not be seamless, i.e. the network definition changes
    '''
    pass


def construct_type(model: nn.Module,
                   type_dict: dict):
    return TYPE_DICT[type_dict.pop('type')].from_dict(type_dict, model)