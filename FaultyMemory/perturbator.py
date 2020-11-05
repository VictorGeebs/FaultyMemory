import torch
import torch.nn as nn
from abc import ABC, abstractclassmethod

# TODO : perturbator : generer un tableau de taille du tenseur + 1 dimension ajoutée avec le sample booleen
# géré en C : concaténer la dim +1 pour obtenir le masque de perturbation `reduce_uint`
# géré en pytorch ou en c: xor, or, and

# TODO: remove and cleanup 
from torch.utils.cpp_extension import load
Cpp_Pert = load(name="Cpp_Pert", sources=["FaultyMemory/cpp/perturbation.cpp"])

PERT_DICT = {}
def add_pert(func):
    PERT_DICT[func.__name__] = func
    return func

class Perturbator(ABC):
    r"""
    Base class for all perturbators that modelize calculation, memory or
    circuit failure.
    """
    repr_compatibility = []
    def __init__(self, **kwargs):
        self._kwargs = {**kwargs}
        self.distribution = {**kwargs}

    def __call__(self, tensor: nn.Tensor):
        if (self.distribution.probs == 0).all():
            return tensor
        if not self.freeze:
            sample = self.distribution.sample(
                sample_shape=tensor.size())
            sample = self.handle_sample(sample, reduce=sample.shape != tensor.shape)
            assert tensor.shape == sample.shape, 'Sampled fault mask shape is not the same as tensor to perturb !'
        else:
            sample = self.saved_sample
        if self.freeze and self.saved_sample is not None:
            self.saved_sample = sample

        shape = tensor.shape
        return self.perturb(tensor.flatten(), sample.flatten()).view(shape)

    @abstractclassmethod
    def handle_sample(self, sample: nn.Tensor, reduce: bool) -> nn.Tensor:
        pass

    @abstractclassmethod
    def define_distribution(self, **kwargs) -> nn.Distribution:
        pass

    @abstractclassmethod
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        r""" How do you apply the perturbation between the `tensor` and `mask`
        """
        pass

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, **kwargs):
        self._distribution = self.define_distribution(**kwargs)
        self.width = len(self.distribution._param)

    def freeze_faults(self):
        self.freeze = True

    def unfreeze_faults(self):
        self.freeze = False
        del self.saved_sample

    def to_json(self):
        dict = {'name': type(self).__name__,
                **self._kwargs}
        return dict

class DigitalPerturbator(Perturbator):
    repr_compatibility = ['DIGITAL']
    def handle_sample(self, sample: nn.Tensor, reduce: bool) -> nn.Tensor:
        if reduce:
            sample.squeeze_(dim=-1) 
            sample = reduce_uint(sample.to(torch.bool))
        return sample.to(torch.uint8)


class AnalogPerturbator(Perturbator):
    repr_compatibility = ['ANALOG']
    def handle_sample(self, sample: nn.Tensor, reduce: bool) -> nn.Tensor:
        if reduce:
            raise ValueError('An analog perturbation should be 1d by definition, the sampled distribution does not follow this principle')
        return sample


class XORPerturbation(DigitalPerturbator):
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        return torch.bitwise_xor(tensor, mask)

class ANDPerturbation(DigitalPerturbator):
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        return torch.bitwise_and(tensor, mask)

class ORPerturbation(DigitalPerturbator):
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        return torch.bitwise_or(tensor, mask)

class AdditiveNoisePerturbation(AnalogPerturbator):
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        return tensor + mask

class MultiplicativeNoisePerturbation(AnalogPerturbator):
    def perturb(self, tensor: nn.Tensor, mask: nn.Tensor) -> nn.Tensor:
        return tensor * mask

@add_pert
class BernoulliXORPerturbation(XORPerturbation):
    def define_distribution(self, **kwargs) -> nn.Distribution:
        return torch.distributions.bernoulli.Bernoulli(**kwargs)


def construct_pert(pert_dict, user_pert = {}):
    r"""
    Constructs a representation according to the dictionnary provided.
    The dictionnary should have a field for 'name' equals to the name of the class, the width of the repr
    """
    if pert_dict is None:
        return None
    all_pert = dict(PERT_DICT, **user_pert)
    return all_pert[pert_dict.pop('name')](**pert_dict)