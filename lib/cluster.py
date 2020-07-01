import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import perturbator as P

class Cluster():
    """
    Deprecated, kept around for activations and hooks
    """
    def __init__(self, perturb=None, models=None, tensors=None, acti=None):
        self.perturb = perturb if perturb is not None else []
        self.models = models if models is not None else []
        self.tensors = tensors if tensors is not None else []
        self.acti = acti if acti is not None else []

        for model in self.models:
            self.extract_tensors(model)

    def __str__(self):
        print("\nperturbs:\n")
        for pert in self.perturb:
            print(pert)
        
        print("\nmodels:\n")
        for model in self.models:
            print(model)
        
        print("\ntensors:\n")
        for tensor in self.tensors:
            print(tensor)
        return ""

    def add_model(self, model):
        r"""
        Adds a model to the list at the specified index, or appends it to the list
        if no index is specified
        """
        self.models.append(model)
        self.extract_tensors(model)

    def remove_model(self, model):
        r"""
        Removes the specified model from the list. This can be an index, and will 
        remove the model at that index of the list.
        """
        if issubclass(type(model), nn.Module):
            try:
                for tensor in list(model.parameters()):
                    self.remove_tensor(tensor)
                self.models.remove(model)
            except ValueError:
                print("Specified model was not found in the list")
            finally:
                pass
        elif type(model) == int:
            del self.models[model]
        else:
            raise TypeError("Type provided was neither an index or a model")

    def add_tensor(self, tensor): #TODO: combine with add_model for cleanliness
        self.tensors.append(tensor)

    def remove_tensor(self, tensor):
        r"""
        Removes the specified tensor from the list. This can be an index, and will 
        remove the tensor at that index of the list.
        """
        if issubclass(type(tensor), torch.Tensor):
            try:
                for i, tens in enumerate(self.tensors):
                    if tens is tensor:
                        print(i)
                        self.tensors.pop(i)
                #self.tensors.remove(tensor)
            except ValueError:
                print("Specified model was not found in the list")
            finally:
                pass
        elif type(tensor) == int:
            del self.tensors[tensor]
        else:
            raise TypeError("Type provided was neither an index or a tensor")

    def extract_tensors(self, model):
        for tensor in list(model.parameters()):
            self.add_tensor(tensor)

    def add_perturb(self, perturb, index=-1):
        r"""
        Adds a perturbation to the list at the specified index, or appends it to the list
        if no index is specified
        """
        self.perturb.insert(index, perturb)
    
    def remove_perturb(self, perturb):
        r"""
        Removes the specified perturbation from the list. This can be an index, and will 
        remove the perturbation at that index of the list.
        """
        if issubclass(type(perturb), P.Perturbator):
            try:
                self.perturb.remove(perturb)
            except ValueError:
                print("Specified perturbation was not found in the list")
            finally:
                pass
        elif type(perturb) == int:
            del self.perturb[perturb]
        else:
            raise TypeError("Type provided was neither an index or a perturbation")

    def perturb_models(self): # Cleanup
        for tensor in self.tensors:
            for perturb in self.perturb:
                perturb(tensor)
        for model in self.models:
            for perturb in self.perturb:
                model.apply_perturb(perturb)

    def contains(self, model): # ADD TENSOR FCT
        """
        Checks if the model passed in parameters is already in this cluster's models
        """
        for mod in self.models:
            if model is mod:
                return True
        return False

    def apply_hooks(self):
        hooks = {}
        for acti in self.acti:
            for i, module in enumerate(self.models):
                hooks[i] = module.register_forward_hook(acti.perturb)


class TensorCluster():
    """
    A faulty memory cluster that stores a collection of tensors to perturb them during the forward pass
    """
    def __init__(self, perturb=None, tensors=None):
        self.perturb = perturb if perturb is not None else []
        self.tensors = tensors if tensors is not None else []
        self.save_tensors()

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
                perturb(tensor)

    def add_tensor(self, tensor):
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
    
    def add_module(self, module):
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

    def contains(self, tensor):
        """
        Verifies if the specified tensor is already in the cluster's memory.
        """
        for tens in self.tensors:
            if tens is tensor:
                return True
        return False

    def save_tensors(self):
        self.saved_tensors = copy.deepcopy(self.tensors)

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

