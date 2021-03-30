"""Tests for representation.py."""

import copy

import FaultyMemory as FyM
import torch
import pytest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

representation = FyM.BinaryRepresentation()


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
            self.feature.weight.zero_().add_(2)

        def forward(self, x):
            return self.feature(x)


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


def test_represented_weight_safe(simple_module):
    rp = FyM.RepresentedParameter(simple_module, "feature.weight", representation)
    ref = copy.deepcopy(simple_module.feature.weight)
    rp.quantize_perturb()
    assert not torch.equal(ref, simple_module.feature.weight)
    shorthand = simple_module.feature.weight.data
    mask = (shorthand == 1) | (shorthand == -1)
    assert mask.sum() == torch.numel(shorthand)
    assert rp.tensor_stats["bitcount"] == torch.numel(shorthand)
    rp.restore()
    assert torch.equal(simple_module.feature.weight, ref)


def test_represented_activation_safe(simple_module, simple_tensor):
    ra = FyM.RepresentedActivation(simple_module, "feature", representation)
    out = simple_module(simple_tensor)
    ra.quantize_perturb()
    out_pert = simple_module(simple_tensor)
    assert not torch.equal(out, out_pert)
    mask = (out_pert == 1) | (out_pert == -1)
    assert mask.sum() == torch.numel(out_pert)
    assert ra.tensor_stats["bitcount"] == torch.numel(out_pert) / 64
    ra.__del__()  # del ra do not delete immediately
    out_init = simple_module(simple_tensor)
    assert torch.equal(out_init, out)


def test_represented_metrics(scalar_module):
    pass
