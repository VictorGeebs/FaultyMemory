import torch
import torch.nn as nn
import random
import numpy as np
import math as math
import warnings
from json import JSONEncoder

from FaultyMemory.representation import *

from typing import Optional

BINARY_SCALING = ['none', 'he', 'mean']

class Perturbator():
    """
    Base class for all perturbators that modelize calculation, memory or 
    circuit failure.
    """
    def __init__(self, p=1.):
        assert (p >= 0. and p <= 1.), "probability p must be between 0 and 1"
        self.p = p

    def __call__(self, params, repr=None, scaling=False): # Make flexible for calling with hooks or with only a tensor? __call__(self, params=None, module=None, in=None, out=None) Naming gets out of wack
        self.perturb(params, repr, scaling)

    def __str__(self):
        return "Base class"

    def __repr__(self):
        return self.__str__

    def perturb(self, params, repr=None, scaling=False):
        r"""
        This function is the transformation that is applied to the data when
        the perturbator is called in  __call__(self, params).
        Should be overridden by all subclasses.
        params should be the parameters that you wish to be modified, by calling
        net.parameters()
        """
        pass

    def set_probability(self, p=1):
        self.p = p

    def hook(self, module, inp, out):
        return self.perturb(out)

    def to_json(self):
        dict = {}
        dict["name"] = self.__class__.__name__
        dict["p"] = self.p
        return dict

class BitwisePert(Perturbator):
    def __init__(self, p: float = 1.):
        assert (p >= 0. and p <= 1.), "probability p must be between 0 and 1"
        self.p = p
    
    def __str__(self):
        return "Bitwise Perturbation"

    def __repr__(self):
        return self.__str__()

    def perturb(self, param: torch.tensor, repr: Optional[Perturbator] = None, scaling: Optional[str] = 'none'):
        param_shape, param_mean = param.shape, torch.mean(torch.abs(param)).item()
        param = param.flatten()
        self.mask = self.generate_tensor_mask_bit(repr.width, param.shape[0])
        data = param.detach().cpu().numpy()
        for i, _ in enumerate(data):
            data[i] = repr.convert_to_repr(data[i])
        data = data.astype('int')
        data = repr.apply_tensor_mask(data, self.mask)
        for i, value in enumerate(data):
            param.data[i] = value

        if scaling != 'none':
            with torch.no_grad():
                if scaling == 'he':
                    he_scaling = math.sqrt(2./(np.prod(param_shape)))
                    param *= he_scaling
                elif scaling == 'mean':
                    param *= param_mean
                else:
                    warnings.warn(f'Scaling is not one of {BINARY_SCALING}', UserWarning)

        param = param.view(param_shape)

    def generate_mask(self, width: int):
        mask = np.zeros(8, dtype=int)
        for i in range(1, width+1):
            if (random.random() <= self.p):
                mask[-i] = 1
        return np.packbits(mask)[0]

    def generate_tensor_mask_bit(self, width: int, tensor_length: int):
        r""" Generate a fault mask the same size as `tensor_length` with precision `width``

        Also set the attribute `effective_p` for bookeeping
        """
        mask = np.random.binomial(1, self.p, (width, tensor_length))
        self.effective_p = np.count_nonzero(mask)/tensor_length
        return np.packbits(mask, axis=0, bitorder='little')[0]

class Zeros(Perturbator):
    """
    A 'Stuck-at-Zero' perturbation, regardless of representation
    """
    def __init__(self, p=1):
        super(Zeros, self).__init__()
        self.p = p

    def __str__(self):
        return "Zero Perturbation"

    def __repr__(self):
        return self.__str__()

    def perturb(self, param, repr=None):
        param_shape = param.shape
        param = param.flatten()
        for i, _ in enumerate(param.data):
            if repr is not None:
                param.data[i] = repr.convert_to_repr(param.data[i])
            if (random.random() <= self.p):
                param.data[i] = 0
        param = param.view(param_shape)

class SignInvert(Perturbator):
    """
    Perturbation that inverts the sign of the input
    """
    def __init__(self, p=1):
        super(SignInvert, self).__init__()
        self.p = p

    def __str__(self):
        return "SignInvert Perturbation"

    def __repr__(self):
        return self.__str__()

    def perturb(self, param, repr=None):
        param_shape = param.shape
        param = param.flatten()
        mask = np.random.binomial(1, self.p, (param.shape[0]))
        data = param.detach().numpy()
        for i, _ in enumerate(data):
            data[i] = repr.convert_to_repr(data[i])
        data = data.astype('int')
        #mask = torch.ones_like(param)
        #for i, _ in enumerate(mask):
        #    if random.random() <= self.p:
        #        mask[i] = 0
        mask = mask*2 - 1
        data *= mask
        for i, value in enumerate(data):
            param.data[i] = value
        param = param.view(param_shape)

class Ones(Perturbator):
    """
    A 'Stuck-at-One' perturbation, regardless of representation
    """
    def __init__(self, p=1):
        super(Ones, self).__init__()
        self.p = p

    def __str__(self):
        return "Ones Perturb"

    def perturb(self, params, repr=None):
        param_shape = param.shape
        param = param.flatten()
        for i, _ in enumerate(param.data):
            if (random.random() <= self.p):
                param.data[i] = 1
        param = param.view(param_shape)

class Gauss(Perturbator):
    """
    Introduces gaussian noise into the inputs
    """
    def __init__(self, p=1, mu=0, sigma=1):
        super(Gauss, self).__init__()
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def __str__(self): # Add mean and std dev to print
        return "Gaussian Perturb"

    def perturb(self, params, repr=None):
        param_shape = param.shape
        param = param.flatten()
        for i, _ in enumerate(param.data):
            if (random.random() <= self.p):
                param.data[i] += random.gauss(self.mu, self.sigma) * param.data[i]
        param = param.view(param_shape)

"""
This dictionnary is used to construct perturbations from a JSON input
"""
PerturbatorDict = {
    "BitwisePert": BitwisePert,
    "Zeros": Zeros,
    "SignInvert": SignInvert,
    "Ones": Ones,
    "Gauss": Gauss
}

def construct_pert(pert_dict):
    """
    Constructs a perturbation according to the dictionnary provided.
    The dictionnary should have a field for 'name' equals to the name of the class and a probability p.
    """
    if pert_dict is None:
        return None
    instance = PerturbatorDict[pert_dict['name']](p=pert_dict['p'])
    return instance
