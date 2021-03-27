import sys

sys.path.append("/home/sebastien/workspace/FaultyMemory")  # FIXME quick and dirty
import FaultyMemory as FyM
import torch
import cProfile
from utils import timefunc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

dummy_tensor = torch.randn([256, 64, 32, 32]).to(device)

slow_fp = FyM.SlowFixedPointRepresentation()
kernel_fp = FyM.FixedPointRepresentation()


def slow_fp_profile():
    slow_fp.quantize(dummy_tensor)


def kernel_fp_profile():
    kernel_fp.quantize(dummy_tensor)


timefunc(slow_fp_profile, iterations=5)
cProfile.run("slow_fp_profile()", sort="tottime")

timefunc(kernel_fp_profile, iterations=5)
cProfile.run("kernel_fp_profile()", sort="tottime")

# TODO additive log to .csv with (datetime, device, func)
