"""Declaring infos."""
import time

__version__ = "0.0.1"
__author__ = "Victor Gaudreau-Blouin, Sébastien Henwood"
__author_email__ = "noreply@polymtl.ca"
__license__ = "Apache-2.0"
__copyright__ = f'Copyright (c) 2020-{time.strftime("%Y")}, {__author__}.' 
__homepage__ = "https://github.com/VictorGeebs/FaultyMemory"

__docs__ = "FaultyMemory is a lightweight wrapper to simulate the effect of hardware architecture on Pytorch models."
__long_docs__ = """
In real-world scenario a deep learning model may encounter unexpected bit-flips.
FaultyMemory aims to brings painless fault-aware training by providing an easy-to-use wrapper that simulates the faults.
The user can describe its model in terms of numerical representations (int, uint, binary, ...).
"""
