import torch
import torch.nn as nn
import random
import numpy as np


# Representations: int, uint, binary (1 bit int)
# 
def weight_to_int(tensor):
    for param in list(tensor):
        param_shape = param.shape
        param = param.flatten()
        #param = torch.round(param)
        for i, _ in enumerate(param.data):    
            param.data[i] = torch.round(param.data[i])
        param = param.view(param_shape)

def int_to_repr(tensor, repr):
    for param in list(tensor):
        param_shape = param.shape
        param = param.flatten()
        for i, _ in enumerate(param.data):
            param.data[i] = repr(param.data[i])
        param = param.view(param_shape)

def generate_mask(width, p):
    mask = np.zeros(8, dtype=int)
    for i, _ in enumerate(mask):
        if (8-i <= width):
            if (random.random() <= p):
                mask[i] = 1
    mask = np.packbits(mask)
    return mask

def tens_to_repr(tensor, width=8, unsigned=True):
    for param in list(tensor):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                param.data[i] = int_to_custom_repr(param.data[i], width, unsigned)
            param = param.view(param_shape)

def int_to_custom_repr(value, width=8, unsigned=True):
    if width == 1:
        value = int(value > 0)
        if unsigned == False and value == 0:
            value = -1
        return value
    if unsigned == True:
        value = value % (pow(2, width))
    else:
        value = (value % (pow(2, width)))
        if (value >= (pow(2, width)/2)):
            value = value - pow(2, width)
    return value

class Representation():
    def __init__(self, width=8, unsigned=True):
        self.width = width
        self.unsigned = unsigned
    
    def convert_to_repr(self, value):
        if self.unsigned == True:
            value = value % (pow(2, self.width))
            return np.uint8(value)
        else:
            value = (value % (pow(2, self.width)))
            if (value >= (pow(2, self.width)/2)):
                value = value - pow(2, self.width)
        return np.int8(value)

    def apply_mask(self, value, mask):
        return value^mask

    def apply_tensor_mask(self, tensor, mask):
        param = np.bitwise_xor(tensor, mask)
        return param

class BinaryRepresentation(Representation):
    def __init__(self, unsigned=False):
        super(BinaryRepresentation, self).__init__()
        self.width = 1
        self.unsigned=unsigned

    def convert_to_repr(self, value):
        if self.unsigned == False:
            if value <= 0:
                value = -1
            else:
                value = 1
        else:
            if value >= 0.5:
                value = 1
            else:
                value = 0
        return value

    def apply_mask(self, value, mask):
        if mask == 0:
            return value
        else:
            return value*-1

    def apply_tensor_mask(self, tensor, mask):
        param = np.bitwise_xor(tensor, mask)
        param = (param * -1) - 1
        return param

class Perturbator():
    """
    Base class for all perturbators that modelize calculation, memory or 
    circuit failure.
    """
    def __init__(self, p=1.):
        assert (p >= 0. and p <= 1.), "probability p must be between 0 and 1"
        self.p = p

    def __call__(self, params, repr=None): # Make flexible for calling with hooks or with only a tensor? __call__(self, params=None, module=None, in=None, out=None) Naming gets out of wack
        self.perturb(params, repr)

    def __str__(self):
        return "Base class"
    def __repr__(self):
        return self.__str__

    def perturb(self, params, repr=None):
        """
        This function is the transformation that is applied to the data when
        the perturbator is called in  __call__(self, params).
        Should be overridden by all subclasses.
        params should be the parameters that you wish to be modified, by calling
        net.parameters()
        """
        pass

    def hook(self, module, inp, out):
        return self.perturb(out)

class BitwisePert(Perturbator):
    def __init__(self, p=1):
        assert (p >= 0. and p <= 1.), "probability p must be between 0 and 1"
        self.p = p
    
    def __str__(self):
        return "Bitwise Perturb"

    def perturb(self, param, repr=None):
        param_shape = param.shape
        param = param.flatten()
        mask = self.generate_tensor_mask_bit(repr.width, param.shape[0])
        data = param.detach().numpy()
        for i, _ in enumerate(data):
            data[i] = repr.convert_to_repr(data[i])
        data = data.astype('int')
        data = repr.apply_tensor_mask(data, mask)
        for i, value in enumerate(data):
            param.data[i] = value
        param = param.view(param_shape)


    def generate_mask(self, width):
        mask = np.zeros(8, dtype=int)
        for i in range(1, width+1):
            if (random.random() <= self.p):
                mask[-i] = 1
        return np.packbits(mask)[0]

    def generate_tensor_mask_bit(self, width, tensor_length):
        mask = np.random.binomial(1, self.p, (width, tensor_length))
        return np.packbits(mask, axis=0, bitorder='little')[0]

    def generate_tensor_mask_int(self, width, tensor_length):
        return np.random.randint(0, 2**width, tensor_length, dtype=np.uint8)

    def weight_to_int(self, tensor):
        for param in list(tensor):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):    
                param.data[i] = torch.round(param.data[i])
            param = param.view(param_shape)

    def tensor_to_repr(self, tensor, repr):
        for param in list(tensor):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                param.data[i] = self.value_to_repr(param.data[i], repr)
            param = param.view(param_shape)

    def value_to_repr(self, value, repr):
        if repr.unsigned == True:
            value = value % (pow(2, repr.width))
            return np.uint8(value)
        else:
            value = (value % (pow(2, repr.width)))
            if (value > (pow(2, repr.width)/2)):
                value = value - pow(2, repr.width)
        return np.int8(value)

class Zeros(Perturbator):
    def __init__(self, p=1):
        super(Zeros, self).__init__()
        self.p = p

    def __str__(self):
        return "Zero Perturb"

    def perturb(self, params, repr=None):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] = 0
            param = param.view(param_shape)

class SignInvert(Perturbator):
    def __init__(self, p=1):
        super(SignInvert, self).__init__()
        self.p = p

    def __str__(self):
        return "SignInvert Perturbation"

    def perturb(self, param, repr=None):
        #for param in list(params):
        param_shape = param.shape
        param = param.flatten()
        mask = torch.ones_like(param)
        for i, _ in enumerate(mask):
            if random.random() <= self.p:
                mask[i] = 0
        mask = mask*2 - 1
        param *= mask
        param = param.view(param_shape)

class Ones(Perturbator):
    def __init__(self, p=1):
        super(Ones, self).__init__()
        self.p = p

    def __str__(self):
        return "Ones Perturb"

    def perturb(self, params, repr=None):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] = 1
            param = param.view(param_shape)

class Twos(Perturbator):
    def __init__(self, p=1):
        super(Twos, self).__init__()
        self.p = p

    def __str__(self):
        return "Twos Perturb"

    def perturb(self, params, repr=None):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] = 2
            param = param.view(param_shape)

class Gauss(Perturbator):
    def __init__(self, p=1, mu=0, sigma=1):
        super(Gauss, self).__init__()
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def __str__(self): # Add mean and std dev to print
        return "Gaussian Perturb"

    def perturb(self, params, repr=None):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] += random.gauss(self.mu, self.sigma) * param.data[i]
            param = param.view(param_shape)
