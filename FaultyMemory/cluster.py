from typing import Callable, List

import torch

from FaultyMemory.perturbator import Perturbator
from FaultyMemory.utils import kmeans_nparray, sanctify_ten, sanitize_number


class Cluster(object):
    """
    A faulty memory cluster that stores a collection of tensors to perturb them during the forward pass

    Args:
        nb_clusters: number of clusters to reduce to (default 0 -> no clustering)
    """

    def __init__(self, nb_clusters: int = 0):
        self.pert = set()
        self.change_nb_clusters(nb_clusters)

    def __str__(self):
        print("Perturbs:")
        for pert in self.perturb:
            print(pert)

    def change_nb_clusters(self, nb_clusters: int = 0):
        nb_clusters = sanitize_number(nb_clusters, mini=0, rnd=True)
        self.nb_clusters = nb_clusters
        if self.pert:
            self.de_cluster()
            self.cluster()

    def assign_perts(self, perturbations: List[Perturbator]):
        self.pert.update(perturbations)
        self.ref_params = torch.stack(
            [pert.distribution._param.view(-1) for pert in self.pert]
        )

    def cluster(self, cluster_func: Callable = kmeans_nparray) -> None:
        r"""Cluster the perturbations held by this object

        Args:
            cluster_func: a callable taking two args: a numpy array (1d) and the number of clusters, returns a numpy array (1d)
        """
        if len(self.pert) == 0:
            raise ValueError("Tried to cluster with no assigned perturbations")
        if self.nb_clusters == 0:
            return

        if not self.saved:
            self.saved = True
            self.saved_params = sanctify_ten(self.ref_params)
        new_assignment = cluster_func(
            np_array=self.ref_params.cpu().numpy(), nb_clusters=self.nb_clusters
        )
        assert new_assignment.shape == self.ref_params.cpu().numpy().shape
        self.ref_params.data.copy_(torch.from_numpy(new_assignment))

    def de_cluster(self):
        if not self.saved:
            print("Nothing to de-cluster !")
        else:
            self.saved = False
            self.ref_params.data.copy_(self.saved_params.data)
