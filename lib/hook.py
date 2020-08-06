from perturbator import *
from representation import *

class Hook():
    def __init__(self, pert, repr):
        self.pert = pert if pert is not None else []
        self.repr = repr
    
    def hook_fn(self, module, input, output):
        for pert in self.pert:
            pert(output, repr=self.repr)
        