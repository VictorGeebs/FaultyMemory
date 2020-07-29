import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import perturbator as P


class Cluster():
    """
    A faulty memory cluster that stores a collection of tensors to perturb them during the forward pass
    """
    def __init__(self, perturb=None, tensors=None, activations=None, networks=None, network_activations=None, repr=None):
        self.perturb = perturb if perturb is not None else []
        self.tensors = tensors if tensors is not None else []
        self.saved_tensors = None
        self.activations = activations if activations is not None else []
        self.repr = repr
        self.hooks = {}
        self.save_tensors()

        if networks is not None:
            for net in networks:
                self.add_module(net)
        if network_activations is not None:
            for net in network_activations:
                self.add_network_activation(net)
        self.apply_hooks()

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
                perturb(tensor, self.repr)

    def restore_tensors(self):
        """
        Restores the saved value for each tensor back to the original tensor.\n
        Tensors are modified in-place, without modifying their reference.
        """
        for param, saved in zip(self.tensors, self.saved_tensors):
            param_shape = param.shape
            saved_shape = saved.shape
            param = param.flatten()
            saved = saved.flatten()
            for i, _ in enumerate(param.data):
                    param.data[i] = saved.data[i]
            param = param.view(param_shape)
            saved = saved.view(saved_shape)

    def save_tensors(self):
        self.saved_tensors = copy.deepcopy(self.tensors)

    def add_tensor(self, tensor, repr=None):
        """
        Adds the specified tensor to the cluster's memory after verifying it isn't already present.
        """
        if self.contains(tensor) == False:
            self.tensors.append(tensor)
            self.saved_tensors.append(copy.deepcopy(tensor))

    def remove_tensor(self, tensor):
        """
        Removes the specified tensor from the cluster's memory and its saved counterpart.
        """
        for i, tens in enumerate(self.tensors):
            if tens is tensor:
                self.tensors.pop(i)
                self.saved_tensors.pop(i)
    
    def add_module(self, module, repr=None):
        """
        Adds every tensor in the specified module (nn.Module) to the cluster's memory with TensorCluster.add_tensor().
        """
        for param in list(module.parameters()):
            self.add_tensor(param)

    def remove_module(self, module):
        """
        Removes every tensor from the specified module (nn.Module) from the cluster's memory and its saved counterpart.
        """
        for tensor in list(module.parameters()):
            self.remove_tensor(tensor)

    def add_activation(self, module, repr=None):
        self.activations.append(module)
        self.apply_hook(module)

    def remove_activation(self, module):
        for i, mod in enumerate(self.activations):
            if mod is module:
                self.clear_hook(module)
                self.activations.pop(i)

    def add_network_activation(self, module):
        for mod in module.children():
            self.add_activation(mod)

    def remove_network_activation(self, module):
        for mod in module.children():
            self.remove_activation(mod)

    def contains(self, tensor):
        """
        Verifies if the specified tensor is already in the cluster's memory.
        """
        for tens in self.tensors:
            if tens is tensor:
                return True
        return False

    def apply_hook(self, module):
        for perturb in self.perturb:
            if (module, perturb) not in self.hooks:
                self.hooks[(module, perturb)] = module.register_forward_hook(perturb.hook)

    def apply_hooks(self):
        for module in self.activations:
            self.apply_hook(module)

    def clear_hook(self, module):
        for perturb in self.perturb:
            if (module, perturb) in self.hooks:           
                self.hooks[(module, perturb)].remove()

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()


