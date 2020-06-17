import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import perturbator as P

class Wrapper(nn.Module):
    def __init__(self, module):
        super(Wrapper, self).__init__()
        self.mod = module
        self.save_model()

    def save_model(self):
        r"""
        Creates a copy of the model parameters to be restored later
        """
        self.saved = copy.deepcopy(self.mod)

    def restore_model(self):
        r"""
        Copies the original saved model back to the used model
        This is used to modify the correct weights during backprop
        """
        self.mod = copy.deepcopy(self.saved)

    def forward(self, x):
        r"""
        An override of the module's Forward(x)
        """
        return self.mod.forward(x)

    def apply_perturb(self, perturb):
        r"""
        Applies the perturbation given to the module's parameters, without 
        regard for weights and biases
        """
        perturb(self.mod.parameters())

def Wrap(module):
    """
    Applies a wrapper to every child in the model. 
    \n NOTE: This could prove to be dangerous as the setattr() method is being used and changes the model itself instead of returning a changed copy.
    \n Possible Fix: use copy.deepcopy() on the model beforehand and returning that changed copy.
    """
    for child in list(module.named_children()):
        child_name = child[0]
        setattr(module, child_name, Wrapper(child[1]))
    return module
