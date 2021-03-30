"""Followed Pytorch-lightning as an example"""

import os
import sys

from setuptools import setup, find_packages

from typing import List

try:
    from FaultyMemory import info
except ImportError:
    # alternative https://stackoverflow.com/a/67692/4521646
    sys.path.append("FaultyMemory")
    import info

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def load_requirements(
    path_dir: str, file_name: str, comment_char: str = "#"
) -> List[str]:
    """Load requirements from a file."""
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


extras = {
    "test": load_requirements(path_dir=_PATH_REQUIRE, file_name="test.txt")
}

setup(
    name="FaultyMemory",
    version=info.__version__,
    description=info.__docs__,
    author=info.__author__,
    author_email=info.__author_email__,
    url=info.__homepage__,
    license=info.__license__,
    keywords=["deep learning", "pytorch", "AI", "hardware emulation"],
    python_requires=">=3.6",
    install_requires=load_requirements(path_dir=_PATH_REQUIRE, file_name="base.txt")+['torch'],
    extras_require=extras,
    packages=find_packages(exclude=["tests", "benchmarks", "benchmarks/*", "tutorial"]),
)
