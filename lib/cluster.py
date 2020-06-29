import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import perturbator as P

class Cluster():
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
                print("\nremoving tensor")
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







class TensorCluster(Cluster):
    def __init__(self, perturb=None, tensors=None, acti=None):
        self.perturb = perturb if perturb is not None else []
        self.tensors = tensors if tensors is not None else []
        self.acti = acti if acti is not None else []

    def __str__(self):
        for pert in self.perturb:
            print(pert)
        for tensor in self.tensors:
            print(tensor)
        return ""

    def add_tensor(self, tensor, index=-1):
        r"""
        Adds a model to the list at the specified index, or appends it to the list
        if no index is specified
        """
        self.tensors.insert(index, tensor)

    def remove_tensor(self, tensor):
        r"""
        Removes the specified model from the list. This can be an index, and will 
        remove the model at that index of the list.
        """
        if issubclass(type(tensor), torch.Tensor):
            try:
                self.tensors.remove(tensor)
            except ValueError:
                print("Specified tensor was not found in the list")
            finally:
                pass
        elif type(tensor) == int:
            del self.tensors[tensor]
        else:
            raise TypeError("Type provided was neither an index or a tensor")

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

    def perturb_tensors(self):
        for tensor in self.tensors:
            for perturb in self.perturb:
                perturb(tensor)

    def contains(self, tensor):
        """
        Checks if the model passed in parameters is already in this cluster's models
        """
        for tens in self.tensors:
            if tensor is tens:
                return True
        return False
