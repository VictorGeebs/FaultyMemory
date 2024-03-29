"""Tests for representation.py."""

import copy

import FaultyMemory as FyM
import torch
import pytest
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

representation = FyM.BinaryRepresentation()


@pytest.fixture
def scalar_tensor() -> torch.Tensor:
    """A simple scalar tensor put on device

    Returns:
        torch.Tensor: scalar tensor of value 1
    """
    return torch.ones([1]).to(device)


@pytest.fixture
def simple_tensor() -> torch.Tensor:
    """A simple 2 dimensional tensor put on device

    Returns:
        torch.Tensor: tensor of symmetric values around 0 from 1 to 3 in (2x3) format
    """
    return torch.randn(64, 16).to(device)


@pytest.fixture
def scalar_module() -> torch.nn.Module:
    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = torch.nn.Linear(1, 1, False)
            self.feature.weight.data.zero_().add_(2)

        def forward(self, x):
            return self.feature(x)

    return SimpleModule()


@pytest.fixture
def simple_module() -> torch.nn.Module:
    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = torch.nn.Linear(16, 16)
            torch.nn.init.kaiming_normal_(self.feature.weight)
            self.feature.bias.data.zero_()

        def forward(self, x):
            return self.feature(x)

    return SimpleModule()


# TODO a module with 2 outputs and test acts


def test_represented_weight_safe(simple_module):
    rp = FyM.RepresentedParameter(simple_module, "feature.weight", representation)
    ref = copy.deepcopy(simple_module.feature.weight)
    rp.quantize_perturb()
    assert not torch.equal(ref, simple_module.feature.weight)
    assert torch.equal(rp.saved_ten, ref)
    shorthand = simple_module.feature.weight.data
    mask = (shorthand == 1) | (shorthand == -1)
    assert mask.sum() == torch.numel(shorthand)
    assert rp.tensor_stats["bitcount"] == torch.numel(shorthand)
    rp.restore()
    assert rp.saved_ten is None
    assert torch.equal(simple_module.feature.weight, ref)


def test_represented_activation_safe(simple_module, simple_tensor):
    ra = FyM.RepresentedActivation(simple_module, "feature", representation)
    out = simple_module(simple_tensor)
    ra.quantize_perturb()
    ra.save()
    out_pert = simple_module(simple_tensor)
    assert not torch.equal(out, out_pert)
    assert torch.equal(ra.saved_ten, out)
    mask = (out_pert == 1) | (out_pert == -1)
    assert mask.sum() == torch.numel(out_pert)
    assert ra.tensor_stats["bitcount"] == torch.numel(out_pert) / 64
    ra.__del__()  # del ra do not delete immediately
    out_init = simple_module(simple_tensor)
    assert torch.equal(out_init, out)
    ra.restore()
    assert ra.saved_ten is None


def test_represented_energy(scalar_module):
    u_representation = FyM.UFixedPointRepresentation(width=3, nb_digits=0)
    for combi in itertools.permutations([0.0, 0.278037, 1.0]):
        perturbation = FyM.BernoulliXORPerturbation(probs=torch.Tensor(combi))
        rp = FyM.RepresentedParameter(
            scalar_module, "feature.weight", u_representation, perturbation
        )
        assert pytest.approx(rp.energy_consumption()[1], 1e-4) == 1.1

    # SCALAR PERT
    perturbation = FyM.BernoulliXORPerturbation(probs=1)
    rp = FyM.RepresentedParameter(
        scalar_module, "feature.weight", u_representation, perturbation
    )
    assert rp.energy_consumption()[1] == 0

    perturbation = FyM.BernoulliXORPerturbation(probs=0)
    rp = FyM.RepresentedParameter(
        scalar_module, "feature.weight", u_representation, perturbation
    )
    assert rp.energy_consumption()[1] == 3

    # NO PERT
    rp = FyM.RepresentedParameter(scalar_module, "feature.weight", u_representation)
    assert rp.energy_consumption()[1] == 3


def test_represented_mse(scalar_module, scalar_tensor):
    u_representation = FyM.UFixedPointRepresentation(width=3, nb_digits=0)
    perturbation = FyM.BernoulliXORPerturbation(probs=torch.Tensor([1.0, 0.0, 1.0]))
    rp = FyM.RepresentedParameter(
        scalar_module, "feature.weight", u_representation, perturbation
    )
    out = scalar_module(scalar_tensor)
    assert out == 2
    rp.quantize_mse()
    rp.quantize_perturb()
    assert scalar_module.feature.weight.data.item() == 7
    out_pert = scalar_module(scalar_tensor)
    assert out_pert != out and out_pert == 7
    assert rp.tensor_stats["MSE"] == 25

    perturbation = FyM.BernoulliXORPerturbation(probs=1)
    rp = FyM.RepresentedParameter(
        scalar_module, "feature.weight", u_representation, perturbation
    )
    assert scalar_module.feature.weight.data.item() == 2
    rp.quantize_mse()
    rp.quantize_perturb()
    assert scalar_module.feature.weight.data.item() == 5
    assert rp.tensor_stats["MSE"] == 9


def test_represented_on_off(scalar_module, scalar_tensor):
    u_representation = FyM.UFixedPointRepresentation(width=3, nb_digits=0)
    perturbation = FyM.BernoulliXORPerturbation(probs=torch.Tensor([1.0, 0.0, 1.0]))
    rp = FyM.RepresentedParameter(
        scalar_module, "feature.weight", u_representation, perturbation
    )
    rp.off_perturbs()
    rp.quantize_perturb()
    out_quantized = scalar_module(scalar_tensor)
    assert out_quantized == 2
    rp.restore()
    rp.on_perturbs()
    rp.quantize_perturb()
    out_perturbed = scalar_module(scalar_tensor)
    assert out_perturbed == 7
