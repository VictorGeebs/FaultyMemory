from typing import List
from scipy.cluster.vq import kmeans, vq, whiten
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import FaultyMemory.perturbator as P
from FaultyMemory.utils import sanctify_ten, sanitize_number


class Cluster(object):
    """
    A faulty memory cluster that stores a collection of tensors to perturb them during the forward pass

    Args:
        nb_clusters: number of clusters to reduce to (default 0 -> no clustering)
    """
    def __init__(self, nb_clusters: int = 0):
        nb_clusters = sanitize_number(nb_clusters, min=0, rnd=True)
        self.nb_clusters = nb_clusters

    def __str__(self):
        print("Perturbs:")
        for pert in self.perturb:
            print(pert)

    def assign_perts(self, perturbations: list):
        self.pert = perturbations
        self.ref_params = torch.stack([pert.distribution._param.view(-1) for pert in self.pert])

    def kmeans(self):
        if not self.saved:
            self.saved = True
            self.saved_params = sanctify_ten(self.ref_params)
        whitened = whiten(self.ref_params.numpy())
        codebook, _ = kmeans(whitened, self.nb_clusters)
        groups, _ = vq(self.ref_params.numpy(), codebook)
        self.ref_params[]