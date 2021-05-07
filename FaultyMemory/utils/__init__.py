"""All the various utils nicely separated per file."""
import os

__all__ = []
for filename in os.listdir(os.path.dirname(__file__)):
    filename = os.path.basename(filename)
    if filename.endswith(".py") and not filename.startswith("__"):
        __all__.append(filename[:-3])

#from . import *  # noqa
# TODO make it work nicely, right now its FyM.utils.Trainer.Trainer() instead of FyM.utils.Trainer()
# See https://softwareengineering.stackexchange.com/questions/418300/python-dynamically-import-modules/418302#418302?newreg=88d12c38a5364a48b00a4bcbc633a0aa