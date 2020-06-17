import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import perturbator as P
import cluster as C
import wrapper as W

class Handler():
    def __init__(self, net, clusters=None):
        self.net = net
        self.clusters = clusters if clusters is not None else []
        self.models = list(self.net.children())

    def __str__(self):
        for model in self.models:
            print(list(model.named_parameters()))
        return ''

    def forward(self, x):
        r"""
        Saves every model, then perturbs every model by cluster, and then makes the forward pass
        """
        self.save_models()
        self.perturb_models()
        return self.net.forward(x)

    def restore_models(self):
        r"""
        Copies the models' saved parameters back to the normal weights to allow for backpropagation
        """
        for model in self.models:
            model.restore_models()

    def init_clusters(self, clusters=None, models=None): # TODO: SPAGHETTI
        """
        Assigns models to specified clusters.\n
        Clusters should contain the list of clusters you wish to assign models to.\n
        Models should be a list of lists of models, the first list being the models to assign to clusters[0] and so on.\n
        The list of models in models[i] will be assigned to clusters[i].\n
        If no models or clusters are specified, it will split all models in order and distribute them across all clusters in the handler equally.
        """
        if models is None:
            if clusters is None:
                clusters = self.clusters   
            models = list(self.net.children())
            nb_clust = len(clusters)
            nb_models = len(models)
            n = math.ceil(nb_models/nb_clust)
            groups = [models[i:i + n] for i in range(0, len(models), n)]  # Splitting the modules in equal groups according to nb of clusters and nb of models
            models = groups

        for i, cluster in enumerate(clusters):
            for model in models[i]:  # Watch out for out of bounds error
                cluster.add_model(model)
        
    def move_model(self, destination_cluster, model): # Add index cluster option?
        """
        Moves a model from its cluster to the destination cluster
        """
        for cluster in self.clusters:
            if cluster.contains(model):
                cluster.remove_model(model)
        destination_cluster.add_model(model)

    def save_models(self):
        for model in self.models:
            model.save_model()

    def perturb_models(self):
        for cluster in self.clusters:
            cluster.perturb_models()

