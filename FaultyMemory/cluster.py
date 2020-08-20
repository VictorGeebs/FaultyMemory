import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import FaultyMemory.perturbator as P


class Cluster():
    """
    A faulty memory cluster that stores a collection of tensors to perturb them during the forward pass
    """
    def __init__(self, perturb=None):
        self.perturb = perturb if perturb is not None else []
        self.tensors = []

    def __str__(self):
        print("Perturbs:")
        for pert in self.perturb:
            print(pert)
        
        print("Tensors:")
        for tensor in self.tensors:
            print(tensor)
        return ""

    def perturb_tensors(self):
        """
        Applies every perturbation specified in this cluster to each of its tensors.\n
        Tensors are modified in-place, without modifying their reference.
        """
        for tensor in self.tensors:
            for perturb in self.perturb:
                perturb(tensor[0], tensor[1])

    def add_tensor(self, tensor, repr=None):
        """
        Adds the specified tensor to the cluster's memory after verifying it isn't already present.
        """
        if self.contains(tensor) == False:
            self.tensors.append((tensor, repr))

    def remove_tensor(self, tensor):
        """
        Removes the specified tensor from the cluster's memory and its saved counterpart.
        """
        for i, tens in enumerate(self.tensors):
            if tens is tensor[0]:
                self.tensors.pop(i)
    
    def add_module(self, module, repr=None):
        """
        Adds every tensor in the specified module (nn.Module) to the cluster's memory with TensorCluster.add_tensor().
        """
        for param in list(module.parameters()):
            self.add_tensor(param, repr)

    def remove_module(self, module):
        """
        Removes every tensor from the specified module (nn.Module) from the cluster's memory and its saved counterpart.
        """
        for tensor in list(module.parameters()):
            self.remove_tensor(tensor)

    def contains(self, tensor):
        """
        Verifies if the specified tensor is already in the cluster's memory.
        """
        for tens in self.tensors:
            if tens is tensor[0]:
                return True
        return False

    def add_perturbation(self, perturb):
        self.perturb.append(perturb)

    def remove_perturbation(self, perturb):
        self.perturb.pop(perturb, None)
        
    def set_perturb_rate(self, pert_rate):
        for i, p in enumerate(pert_rate):
            self.perturb[i].set_perturb_rate(p)
    
    def set_perturb(self, pert_list):
        self.perturb = pert_list