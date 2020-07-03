import torch
import torch.nn as nn
import random

class Perturbator():
    """
    Base class for all perturbators that modelize calculation, memory or 
    circuit failure.
    """
    def __init__(self, p=1.):
        assert (p >= 0. and p <= 1.), "probability p must be between 0 and 1"
        self.p = p

    def __call__(self, params): # Make flexible for calling with hooks or with only a tensor? __call__(self, params=None, module=None, in=None, out=None) Naming gets out of wack
        self.perturb(params)

    def __str__(self):
        return "Base class"
    def __repr__(self):
        return self.__str__

    def perturb(self, params):
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



class Zeros(Perturbator):
    def __init__(self, p=1):
        super(Zeros, self).__init__()
        self.p = p

    def __str__(self):
        return "Zero Perturb"

    def perturb(self, params):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] = 0
            param = param.view(param_shape)
            
class Ones(Perturbator):
    def __init__(self, p=1):
        super(Ones, self).__init__()
        self.p = p

    def __str__(self):
        return "Ones Perturb"

    def perturb(self, params):
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

    def perturb(self, params):
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

    def perturb(self, params):
        for param in list(params):
            param_shape = param.shape
            param = param.flatten()
            for i, _ in enumerate(param.data):
                if (random.random() <= self.p):
                    param.data[i] += random.gauss(self.mu, self.sigma) * param.data[i]
            param = param.view(param_shape)
