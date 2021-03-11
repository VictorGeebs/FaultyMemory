from setuptools import setup, find_packages

import FaultyMemory

setup(
    name="FaultyMemory",
    version=FaultyMemory.__version__,
    description=FaultyMemory.__docs__,
    author=FaultyMemory.__author__,
    author_email=FaultyMemory.__author_email__,
    url=FaultyMemory.__homepage__,
    license=FaultyMemory.__license__,
    keywords=["deep learning", "pytorch", "AI", "hardware emulation"],
    python_requires=">=3.6",
)
