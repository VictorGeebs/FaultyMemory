import pytest
import torch
import numpy as np
import torch.nn as nn
import copy

from FaultyMemory.representation import BinaryRepresentation
from FaultyMemory.handler import Handler
from FaultyMemory.perturbator import BitwisePert


def test_binary_faults():   #TODO DRY 
    class dummy_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('dummy', nn.Parameter(torch.empty(10).normal_(0,3)))

        def forward(self, x):
            return x * self.dummy

    dummy = dummy_module()
    bin_repr = BinaryRepresentation(width=1, unsigned=False)
    handler = Handler(dummy)

    pert = BitwisePert(p=0.)
    handler.add_tensor('dummy', representation=bin_repr, perturb=[pert])
    handler.perturb_tensors()

    # no faults
    perturbed_ten = copy.deepcopy(dummy.dummy.detach().numpy())
    assert np.isin(perturbed_ten, [1, -1]).all()

    handler.restore()
    handler.remove_tensor('dummy')
    pert = BitwisePert()
    handler.add_tensor('dummy', representation=bin_repr, perturb=[pert])
    handler.perturb_tensors()
    
    print(perturbed_ten)
    # faults everywhere : should sum to 0 (-1+1 for each slot of the tensor)
    perturbed_ten_inv = copy.deepcopy(dummy.dummy.detach().numpy())
    print(perturbed_ten_inv)
    assert sum(perturbed_ten + perturbed_ten_inv) == 0


def test_binary_representation():
    class dummy_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('dummy', nn.Parameter(torch.empty(10).normal_(0,3)))

        def forward(self, x):
            return x * self.dummy

    dummy = dummy_module()
    bin_repr = BinaryRepresentation(width=1, unsigned=False)
    handler = Handler(dummy)
    handler.add_tensor('dummy', representation=bin_repr)

    # original tensor
    clean_ten = copy.deepcopy(dummy.dummy.detach().numpy())

    # perturbed tensor
    handler.perturb_tensors()
    perturbed_ten = copy.deepcopy(dummy.dummy.detach().numpy())
    assert all(x in [-1, 1] for x in perturbed_ten)

    # The element-wise product of the binary representation and original value
    # should give positive values (negative times negative -> positive)
    print((clean_ten * perturbed_ten))
    assert all(x >= 0 for x in (clean_ten * perturbed_ten))

    # Unsigned = True
    dummy = dummy_module()
    bin_repr = BinaryRepresentation(width=1, unsigned=True)
    handler = Handler(dummy)
    handler.add_tensor('dummy', representation=bin_repr)

    # original tensor
    clean_ten = copy.deepcopy(dummy.dummy.detach().numpy())

    # perturbed tensor
    handler.perturb_tensors()
    perturbed_ten = copy.deepcopy(dummy.dummy.detach().numpy())
    assert all(x in [0, 1] for x in perturbed_ten)

    # The element-wise product of the binary representation and original value
    # should give positive values (negative times negative -> positive)
    assert all(x >= 0 for x in (clean_ten * perturbed_ten))
