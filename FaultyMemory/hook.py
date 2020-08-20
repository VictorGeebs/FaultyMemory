from FaultyMemory.perturbator import *
from FaultyMemory.representation import *

class Hook():
    """
    An extention of pytorch hooks that allow for perturbations and representations
    """
    def __init__(self, pert, repr):
        self.pert = pert if pert is not None else []
        self.repr = repr
    
    def hook_fn(self, module, input, output):
        """
        Hooking this function onto a module will perturbate the output with its perturbations and representation.
        """
        for pert in self.pert:
            pert(output, repr=self.repr)
        