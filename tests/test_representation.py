import pytest
import torch
import numpy as np
import torch.nn as nn

from FaultyMemory.representation import BinaryRepresentation
from FaultyMemory.handler import Handler

def test_binary_faults():
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
    handler.perturb_tensors()

    perturbed_ten = dummy.dummy.detach().numpy()
    assert (perturbed_ten == -1).all() or (perturbed_ten == 1).all()