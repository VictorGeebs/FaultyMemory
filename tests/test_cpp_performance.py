import sys, os
from tqdm import tqdm
import json
import time
import FaultyMemory.utils as utils
import FaultyMemory.handler as H
import FaultyMemory.cluster as C
import FaultyMemory.perturbator as P
import torch
import numpy as np
import copy

from torch.utils.cpp_extension import load

Cpp_Pert = load(name="Cpp_Pert", sources=["FaultyMemory/cpp/perturbation.cpp"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


width = 3

vec = [x for x in range(-5, 5, 1)]
print("vec: ", vec)
ten_raw = torch.FloatTensor(vec)
ten_py = copy.deepcopy(ten_raw)

repr_vec = [Cpp_Pert.reprSetWidth(val, width, True) for val in vec]
print("repr_vec: ", repr_vec)
ten_repr = torch.FloatTensor(repr_vec)

Cpp_Pert.applyPerturbMask(ten_raw, width, 0.5)
print()
Cpp_Pert.applyPerturbMask(ten_repr, width, 0.5)
print("ten_raw: ", ten_raw)
print("ten_repr: ", ten_repr)


def test_cpp_perturb(tensors):
    pass


if __name__ == '__main__':
    pass
