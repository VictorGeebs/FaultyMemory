"""Root package info."""

from FaultyMemory.info import (  # noqa: F401
    __author__,
    __author_email__,
    __copyright__,
    __docs__,
    __homepage__,
    __license__,
    __version__,
)

from .handler import *  # noqa: E402 F401
from .cluster import *  # noqa: E402 F401
from .perturbator import *  # noqa: E402 F401
from .representation import *  # noqa: E402 F401
from .represented_tensor import *  # noqa: E402 F401
from .utils import *  # noqa: E402 F401

__import__("pkg_resources").declare_namespace(__name__)
