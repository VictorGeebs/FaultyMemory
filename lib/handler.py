import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import perturbator as P
import representation as R
from hook import *
import cluster as C
import time


class Handler():
    def __init__(self, net, clusters=None):
        self.net = net
        self.saved_net = copy.deepcopy(net)
        self.clusters = clusters if clusters is not None else []
        self.modules = list(self.net.children())
        self.tensor_info = {}
        self.acti_info = {}
        self.hooks = {}

    def __str__(self):
        print("Handler: ")
        for cluster in self.clusters:
            print(cluster)
        return ''

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        r"""
        Saves every module, then perturbs every module by cluster, and then makes the forward pass
        """
        print("saving modules")
        self.save_modules()
        print("Perturbing modules")
        start_time = time.time()
        #self.perturb_modules()
        self.perturb_tensors()
        tot_time = time.time()-start_time
        print("Time to perturb: ", tot_time)
        print("making fwd pass")
        start_time = time.time()
        out =  self.net.forward(x)
        tot_time = time.time()-start_time
        print("Time to fwd pass: ", tot_time)
        return out

    def restore_modules(self):
        r"""
        Copies the modules' saved parameters back to the normal weights to allow for backpropagation
        """
        pert_params = list(self.net.parameters())
        saved_params = list(self.saved_net.parameters())
        for perturbed, saved in zip(pert_params, saved_params):
            perturbed_shape = perturbed.shape
            saved_shape = saved.shape
            perturbed = perturbed.flatten()
            saved = saved.flatten()
            for i, _ in enumerate(perturbed.data):
                    perturbed.data[i] = saved.data[i]
            perturbed = perturbed.view(perturbed_shape)
            saved = saved.view(saved_shape)

        #for cluster in self.clusters:
        #    cluster.restore_tensors()

    def init_clusters(self, clusters=None, modules=None): # TODO: SPAGHETTI
        """
        Assigns modules to specified clusters.\n
        Clusters should contain the list of clusters you wish to assign modules to.\n
        Modules should be a list of lists of modules, the first list being the modules to assign to clusters[0] and so on.\n
        The list of modules in modules[i] will be assigned to clusters[i].\n
        If no modules or clusters are specified, it will split all modules in order and distribute them across all clusters in the handler equally.
        """
        if modules is None:
            if clusters is None:
                clusters = self.clusters   
            modules = list(self.net.children())
            nb_clust = len(clusters)
            nb_modules = len(modules)
            n = math.ceil(nb_modules/nb_clust)
            groups = [modules[i:i + n] for i in range(0, len(modules), n)]  # Splitting the modules in equal groups according to nb of clusters and nb of modules
            modules = groups

        for i, cluster in enumerate(clusters):
            for module in modules[i]:  # Watch out for out of bounds error
                cluster.add_module(module)
        
    def move_tensor(self, destination_cluster, tensor):
        """
        Moves a tensor from its cluster to the destination cluster
        """
        for cluster in self.clusters:
            if cluster.contains(tensor):
                cluster.remove_tensor(tensor)
        destination_cluster.add_tensor(tensor)

    def move_module(self, destination_cluster, module):
        """
        Moves a module from its cluster to the destination cluster
        """
        for cluster in self.clusters:
            cluster.remove_module(module)
        destination_cluster.add_module(module)

    def move_activation(self, destination_cluster, module):
        for cluster in self.clusters:
            cluster.remove_activation(module)
        destination_cluster.add_activation(module)

    def save_modules(self):
        for cluster in self.clusters:
            cluster.save_tensors()

    def perturb_modules(self):
        for cluster in self.clusters:
            cluster.perturb_tensors()

    def perturb_tensors(self):
        for name, item in self.tensor_info.items():
            pert = item[0]
            repr = item[1]
            tens = dict(self.net.named_parameters())[name]
            for perturb in pert:
                perturb(tens, repr)

    def apply_hooks(self):
        for cluster in self.clusters:
            cluster.apply_hooks()

    def from_json(self, handler_dict):
        net_path = handler_dict['net_path']
        nb_clusters = handler_dict['nb_clusters']
        while len(self.clusters) < nb_clusters:
            self.clusters.append(C.Cluster())

        # Weights
        weight_dict = handler_dict['weights']
        if weight_dict is not None:
            # Network batch
            net = weight_dict['net']
            if net is not None:
                repr_dict = net['repr']
                repr = R.construct_repr(repr_dict)

                pert_list = net['perturb']
                perturbs = []
                for pert_dict in pert_list:
                    pert = P.construct_pert(pert_dict)
                    perturbs.append(pert)
                
                for param in list(self.net.named_parameters()):
                    tensor_name = param[0]
                    self.tensor_info[tensor_name] = (perturbs, repr)

            # Modules batch
            modules = weight_dict['modules']
            if modules is not None:
                for module in modules:
                    module_name = module['name']

                    repr_dict = module['repr']
                    repr = R.construct_repr(repr_dict)

                    pert_list = module['perturb']
                    perturbs = []
                    for pert_dict in pert_list:
                        pert = P.construct_pert(pert_dict)
                        perturbs.append(pert)
                    
                    current_mod = dict(self.net.named_modules())[module_name]
                    for param_key in dict(current_mod.named_parameters()):
                        full_key = module_name + '.' + param_key
                        self.tensor_info[full_key] = (perturbs, repr)
                        
            # Tensors
            tensors = weight_dict['tensors']
            if tensors is not None:
                for tensor in tensors:
                    tensor_name = tensor['name']

                    repr_dict = tensor['repr']
                    repr = R.construct_repr(repr_dict)

                    pert_list = tensor['perturb']
                    perturbs = []
                    for pert_dict in pert_list:
                        pert = P.construct_pert(pert_dict)
                        perturbs.append(pert)

                    self.tensor_info[tensor_name] = (perturbs, repr)

        # Activations
        acti_dict = handler_dict['activations']
        if acti_dict is not None:
            # Network batch
            net = acti_dict['net']
            if net is not None:
                repr_dict = net['repr']
                repr = R.construct_repr(repr_dict)

                pert_list = net['perturb']
                perturbs = []
                for pert_dict in pert_list:
                    pert = P.construct_pert(pert_dict)
                    perturbs.append(pert)

                for name, module in self.net.named_modules():
                    hook = Hook(perturbs, repr)
                    self.hooks[name] = module.register_forward_hook(hook.hook_fn)
                    self.acti_info[name] = (perturbs, repr)

            # Modules batch
            modules = acti_dict['modules']
            if modules is not None:
                for module in modules:
                    module_name = module['name']

                    repr_dict = module['repr']
                    repr = R.construct_repr(repr_dict)

                    pert_list = module['perturb']
                    perturbs = []
                    for pert_dict in pert_list:
                        pert = P.construct_pert(pert_dict)
                        perturbs.append(pert)
                    
                    current_mod = dict(self.net.named_modules())[module_name]
                    hook = Hook(perturbs, repr)
                    self.hooks[module_name] = current_mod.register_forward_hook(hook.hook_fn)
                    self.acti_info[module_name] = (perturbs, repr)
