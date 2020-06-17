import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import perturbator as P

class Cluster():
    def __init__(self, perturb=None, models=None):
        self.perturb = perturb if perturb is not None else []
        self.models = models if models is not None else []

    def __str__(self):
        for pert in self.perturb:
            print(pert)
        for model in self.models:
            print(model)
        return ""

    def add_model(self, model, index=-1):
        r"""
        Adds a model to the list at the specified index, or appends it to the list
        if no index is specified
        """
        self.models.insert(index, model)

    def remove_model(self, model):
        r"""
        Removes the specified model from the list. This can be an index, and will 
        remove the model at that index of the list.
        """
        if issubclass(type(model), nn.Module):
            try:
                self.models.remove(model)
            except ValueError:
                print("Specified model was not found in the list")
            finally:
                pass
        elif type(model) == int:
            del self.models[model]
        else:
            raise TypeError("Type provided was neither an index or a model")

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

    def perturb_models(self):
        for model in self.models:
            for perturb in self.perturb:
                model.apply_perturb(perturb)

    def contains(self, model):
        """
        Checks if the model passed in parameters is already in this cluster's models
        """
        for mod in self.models:
            if model is mod:
                return True
        return False