import os
import sys

from setuptools import setup, find_packages

try:
    from FaultyMemory import info
except ImportError:
    # alternative https://stackoverflow.com/a/67692/4521646
    sys.path.append("FaultyMemory")
    import info

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def load_requirements(path_dir: str, file_name: str):
    with open(f"{path_dir}/{file_name}") as f:
        required = f.read().splitlines()
    return required


extras = {
    "test": load_requirements(path_dir=_PATH_REQUIRE, file_name="test.txt"),
    "dev": load_requirements(path_dir=_PATH_REQUIRE, file_name="dev.txt"),
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
    install_requires=load_requirements(path_dir=_PATH_REQUIRE, file_name="base.txt"),
    extras_require=extras,
    packages=find_packages(exclude=["tests", "benchmarks", "benchmarks/*", "tutorial"]),
)
