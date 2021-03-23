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
Cpp_Repr = load(name="Cpp_Repr", sources=["FaultyMemory/cpp/representation.cpp"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perturb(ten: torch.tensor, repr_width, p):
    # TODO smart kernel for GPU/CPU vs multiplication to concatenate mask
    # bitrank vector
    # compact_sample = torch.sum(compact_sample * bitrank, dim=-1)

    ten_repr = Cpp_Pert.generateTensorMask(ten, repr_width, p)

    print(ten_repr)

    ten_np = ten_repr.numpy()
    packed = np.packbits(
        ten_np.astype(int), axis=-1, bitorder="little"
    )  # Packing bits in order to create the mask
    ten_packed = torch.from_numpy(packed)
    mask = torch.flatten(
        ten_packed, start_dim=-2
    )  # Removing the extra dimension caused by the bit packing

    ten.data = torch.bitwise_xor(ten, mask).data


# ---------Perturb Test---------------
# width = 3
# vec = [[x for x in range(0,3)] for y in range(0,3)]
width = 3
p = 0.5

vec = [[x for x in range(0, 3)] for y in range(0, 2)]
ten_raw = torch.ByteTensor(vec)

print(ten_raw)
out = perturb(ten_raw, width, p)


# --------------Encode - Decode Tests------------------

vec = [x / 5 for x in range(-5, 5)]
ten_raw = torch.FloatTensor(vec)
isSigned = True
width = 8
nbDigits = 8
# encoded = Cpp_Repr.encodeTenFixedPoint(ten_raw, width, nbDigits)
vec = [x for x in range(256)]
encoded = torch.FloatTensor(vec)
decoded = Cpp_Repr.decodeTenFixedPoint(encoded, width, nbDigits)
quantized = Cpp_Repr.quantizeTenFixedPoint(ten_raw, width, nbDigits)
# print(ten_raw)
# print(encoded)
# print(decoded)
print("min: ", min(decoded))
print("max: ", max(decoded))
# print(quantized)


# vec = [x/2 for x in range(-30, 30)]
# for val in vec:
#     width = 5
#     isSigned = True
#     nbDigits = 3
#     encoded = Cpp_Repr.encodeBinary(val, isSigned)
#     decoded = Cpp_Repr.decodeBinary(encoded, isSigned)
#     quantized = Cpp_Repr.quantizeBinary(val, isSigned)
#     print("Value: ", val)
#     print("Encoded: ", encoded)
#     print("Decoded: ", decoded)
#     print("Quantize: ", quantized)
#     print("")


def test_cpp_perturb(tensors):
    pass


if __name__ == "__main__":
    pass
