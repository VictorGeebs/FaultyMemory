import torch
import torch.nn as nn
from abc import ABC, abstractclassmethod

# TODO: remove and cleanup
from torch.utils.cpp_extension import load

Cpp_Pert = load(name="Cpp_Pert", sources=["FaultyMemory/cpp/perturbation.cpp"])

BITRANK = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128])

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
        self.distribution = kwargs
        self.freeze = False

    def __call__(self, tensor: torch.Tensor):
        # FIXME if scalar dist but multibits repr, should inflate dist to match repr
        if (self.distribution.probs == 0).all():
            return tensor
        if not self.freeze:
            sample = self.distribution.sample(sample_shape=tensor.size())
            sample = self.handle_sample(sample, reduce=(sample.shape != tensor.shape))
            assert (
                tensor.shape == sample.shape
            ), "Sampled fault mask shape is not the same as tensor to perturb !"
        else:
            sample = self.saved_sample
        if self.freeze and self.saved_sample is not None:
            self.saved_sample = sample

        shape = tensor.shape
        return self.perturb(tensor.flatten(), sample.flatten()).view(shape)

    @abstractclassmethod
    def handle_sample(self, sample: torch.Tensor, reduce: bool) -> torch.Tensor:
        pass

    @abstractclassmethod
    def define_distribution(self, kwargs) -> torch.distributions:
        pass

    @abstractclassmethod
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""How do you apply the perturbation between the `tensor` and `mask`"""
        pass

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, kwargs):
        self._distribution = self.define_distribution(kwargs)
        if self.distribution._param.dim() == 0:  # 0-dim tensor (scalar)
            self.width = 1
        else:
            self.width = len(self.distribution._param)

    def freeze_faults(self):
        self.freeze = True

    def unfreeze_faults(self):
        self.freeze = False
        if self.saved_sample is not None:
            del self.saved_sample

    def to_json(self):
        dict = {"name": type(self).__name__, **self._kwargs}
        return dict


class DigitalPerturbator(Perturbator):
    repr_compatibility = ["DIGITAL"]

    def handle_sample(self, sample: torch.Tensor, reduce: bool) -> torch.Tensor:
        if reduce:
            sample.squeeze_(dim=-1)
            sample = torch.sum(
                sample.to(torch.bool) * BITRANK[: sample.shape[-1]], dim=-1
            )
        return sample.to(torch.uint8)


class AnalogPerturbator(Perturbator):
    repr_compatibility = ["ANALOG"]

    def handle_sample(self, sample: torch.Tensor, reduce: bool) -> torch.Tensor:
        if reduce:
            raise ValueError(
                "An analog perturbation should be 1d by definition, the sampled distribution does not follow this principle"
            )
        return sample


class XORPerturbation(DigitalPerturbator):
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(tensor, mask)


class ANDPerturbation(DigitalPerturbator):
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(tensor, mask)


class ORPerturbation(DigitalPerturbator):
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_or(tensor, mask)


class AdditiveNoisePerturbation(AnalogPerturbator):
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tensor + mask


class MultiplicativeNoisePerturbation(AnalogPerturbator):
    def perturb(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tensor * mask


@add_pert
class BernoulliXORPerturbation(XORPerturbation):
    def define_distribution(self, kwargs) -> torch.distributions:
        return torch.distributions.bernoulli.Bernoulli(**kwargs)


def construct_pert(pert_dict, user_pert={}):
    r"""
    Constructs a representation according to the dictionnary provided.
    The dictionnary should have a field for 'name' equals to the name of the class, the width of the repr
    """
    if pert_dict is None:
        return None
    all_pert = dict(PERT_DICT, **user_pert)
    return all_pert[pert_dict.pop("name")](**pert_dict)
