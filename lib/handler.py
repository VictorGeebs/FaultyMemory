import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import perturbator as P
import cluster as C

class Handler():
    def __init__(self, net, clusters=None):
        self.net = net
        self.clusters = clusters if clusters is not None else []
        self.modules = list(self.net.children())

    def __str__(self):
        print("Handler: ")
        for cluster in self.clusters:
            print(cluster)
        return ''

    def forward(self, x):
        r"""
        Saves every module, then perturbs every module by cluster, and then makes the forward pass
        """
        self.save_modules()
        self.perturb_modules()
        return self.net.forward(x)

    def restore_modules(self):
        r"""
        Copies the modules' saved parameters back to the normal weights to allow for backpropagation
        """
        for cluster in self.clusters:
            cluster.restore_tensors()

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

    def apply_hooks(self):
        for cluster in self.clusters:
            cluster.apply_hooks()