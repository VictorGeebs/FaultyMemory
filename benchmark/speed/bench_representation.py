import sys

sys.path.append("/home/sebastien/workspace/FaultyMemory")  # FIXME quick and dirty
import FaultyMemory as FyM
import torch
import cProfile
from utils import timefunc, profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

dummy_tensor = torch.randn([256, 64, 32, 32]).to(device)

slow_fp = FyM.SlowFixedPointRepresentation()
kernel_fp = FyM.FixedPointRepresentation()
slow_ufp = FyM.USlowFixedPointRepresentation()
kernel_ufp = FyM.UFixedPointRepresentation()


@profile
def slow_fp_profile():
    slow_fp.quantize(dummy_tensor)


@profile
def kernel_fp_profile():
    kernel_fp.quantize(dummy_tensor)


@profile
def slow_ufp_profile():
    slow_ufp.quantize(dummy_tensor)


@profile
def kernel_ufp_profile():
    kernel_ufp.quantize(dummy_tensor)


# TODO additive log to .csv with (datetime, device, func)
