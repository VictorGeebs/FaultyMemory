import json
import torch
import logging
import multiprocessing as mp

from tabulate import tabulate

from FaultyMemory.cluster import Cluster
from FaultyMemory.perturbator import Perturbator
from FaultyMemory.representation import FreebieQuantization, Representation
from FaultyMemory.represented_tensor import (
    RepresentedParameter,
    RepresentedActivation,
    construct_type,
)
from FaultyMemory.utils.misc import ten_exists
from FaultyMemory.utils.Transfer import change_model_output
from FaultyMemory.utils.log_hparams import OUTPUT_SIZE_ALIAS

from typing import Tuple, Union, Optional, Dict


logger = logging.getLogger(__name__)


class Handler:
    r"""
    Class in charge of saving tensors, storing information about them,
    activation perturbations and clusters.
    """

    def __init__(
        self, net: torch.nn.Module, clusters: Optional[int] = 0, param_dict: dict = None
    ):
        """A manager for neural net quantization on faulty hardware.

        Args:
            net (torch.nn.Module): the target net
            clusters (Optional[int], optional): [description]. Defaults to 0.
            param_dict (dict, optional): [description]. Defaults to None.
        """
        self.net = net
        self.represented_ten = {}
        self.add_net_parameters(FreebieQuantization())
        self.add_net_activations(
            FreebieQuantization()
        )  # TODO a method to pick the fused activations
        if param_dict is not None:
            self.from_dict(param_dict)
        self.clusters = Cluster(clusters)

    def __str__(self):  # pragma: no cover
        print("Handler: ")
        for ten in self.represented_ten:
            print(ten)
        return ""

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, grad_enabled: bool = True):
        r"""
        Saves every tensor, then perturbs every tensor, and then makes the
        forward pass

        Args:
            x: Input to process
        """
        with torch.set_grad_enabled(grad_enabled):
            self.perturb_tensors()
            out = self.net.forward(x)
            self.restore()
            return out

    def perturb_tensors(self) -> None:
        [
            represented_ten.quantize_perturb()
            for _, represented_ten in self.represented_ten.items()
        ]

    def restore(self, purge: bool = True) -> None:
        [
            represented_ten.restore(purge)
            for _, represented_ten in self.represented_ten.items()
        ]

    def compute_mse(self) -> None:
        [
            represented_ten.quantize_MSE()
            for _, represented_ten in self.represented_ten.items()
        ]

    def get_stat(self, stat: str) -> list:
        return [
            represented_ten.tensor_stats[stat]
            for represented_ten in self.represented_ten.values()
        ]

    def get_names(self) -> list:
        return [
            represented_ten.name for represented_ten in self.represented_ten.values()
        ]

    def apply(self, func):
        [func(represented_ten) for represented_ten in self.represented_ten.values()]

    def compute_comparative_mse(self, data):
        r"""Compute the MSE with respect to the non-quantized and non-perturbed network for the input tensor `data`."""
        headers = ["Name", "Ref (==0)", "Quantized", "Quantized_perturbed"]
        # Remove quantize_perturb from callbacks if here
        self.apply(lambda repr_ten: repr_ten.detach_callback("quantize_perturb"))
        # First pass: MSE cb, for restore purge=False
        self.apply(lambda repr_ten: repr_ten.quantize_MSE())
        # needed for params
        self.apply(lambda repr_ten: repr_ten.default_exec_callback_stack())
        self.net.forward(data)
        ref = self.get_stat("MSE")
        self.restore(purge=False)

        # Second pass: add quantize_perturb, for restore purge=False
        self.apply(lambda repr_ten: repr_ten.off_perturbs())
        self.perturb_tensors()
        # needed for params
        self.apply(lambda repr_ten: repr_ten.default_exec_callback_stack())
        self.net.forward(data)
        quant = self.get_stat("MSE")
        self.restore(purge=False)

        # Third pass: activate perturbs, for restore purge=True
        self.apply(lambda repr_ten: repr_ten.on_perturbs())
        # needed for params
        self.apply(lambda repr_ten: repr_ten.default_exec_callback_stack())
        self.net.forward(data)
        pert = self.get_stat("MSE")
        self.restore(purge=True)

        # Pretty print results
        tabulate(zip(self.get_names(), ref, quant, pert), headers)

    def value_range(self) -> None:
        [
            represented_ten.value_range()
            for _, represented_ten in self.represented_ten.items()
        ]

    def assign_representation_range(self) -> None:
        [
            represented_ten.adjust_fixed_point()
            for _, represented_ten in self.represented_ten.items()
        ]

    def dnn_wizard(self):
        r"""Parse a neural network and cast it to a best bet quantization + perturbation"""
        # 1 - Fusion known linearities

        # 2 - Cast fused/unfused parameters/acts to represented tensors

        # 2a - Pick repr

        # 2b - Pick pert
        pass

    def parallel_evaluate(
        self, repeat: int, tensor: torch.Tensor, loss: torch.nn.Module, num_workers=4
    ) -> float:
        r"""Massively parallelize a neural network for evaluation of the `tensor` on multiple CPU cores.
        TODO: also use GPU if available
        """
        self.net.to("cpu")
        assert (
            num_workers <= mp.cpu_count()
        ), "Cannot instantiate more workers than CPU cores (I guess it would be useless?)"
        with mp.Pool(num_workers) as p:
            res = [p.apply_async(self.forward, tensor, False) for _ in range(repeat)]
        res = [loss(r) for r in res]

    def add_parameter(
        self,
        name: str,
        representation: Representation,
        perturb: Optional[Union[Dict, Perturbator]] = None,
    ) -> None:
        if name in self.represented_ten:
            logger.info(f"{name} already saved, replacing it")
        self.represented_ten[name] = RepresentedParameter(
            self.net, name, representation, perturb
        )

    def add_activation(
        self,
        name: str,
        representation: Representation,
        perturb: Optional[Union[Dict, Perturbator]] = None,
    ):
        if name in self.represented_ten:
            logger.info(f"{name} already saved, replacing it")
        self.represented_ten[name] = RepresentedActivation(
            self.net, name, representation, perturb
        )

    def remove_tensor(self, name):
        self.represented_ten.pop(name, None)

    def add_module_parameters(
        self,
        name: str,
        representation: Representation,
        perturb: Optional[Union[Dict, Perturbator]] = None,
    ):
        net_dict = self.net.named_modules()
        ten_exists(net_dict, name)
        module = net_dict[name]
        [
            self.add_parameter(f"{name}.{param_key}", representation, perturb)
            for param_key, _ in module.named_parameters()
        ]

    def remove_module_parameters(self, name: str):
        module = self.net.named_modules()[name]
        [
            self.remove_tensor(f"{name}.{param_key}")
            for param_key, _ in module.named_parameters()
        ]

    def add_net_parameters(
        self,
        representation: Representation,
        perturb: Optional[Union[Dict, Perturbator]] = None,
    ):
        _ = [
            self.add_parameter(param_key, representation, perturb)
            for param_key, _ in self.net.named_parameters()
        ]

    def remove_net_parameters(self) -> None:
        _ = [
            self.remove_tensor(param_key)
            for param_key, _ in self.net.named_parameters()
        ]

    def add_net_activations(
        self,
        representation: Representation,
        perturb: Optional[Union[Dict, Perturbator]] = None,
    ) -> None:
        _ = [
            self.add_activation(module, representation, perturb)
            for module, _ in self.net.named_modules()
        ]

    def remove_net_activations(self):
        _ = [self.remove_tensor(module) for module, _ in self.net.named_modules()]

    def purge(self):
        self.represented_ten.clear()

    def from_json(self, file_path):
        """
        Creates a handler dictionnary from a json file and initializes the handler to that configuration
        """
        self.from_dict(Handler.dict_from_json(file_path))

    @staticmethod
    def dict_from_json(file_path: str):
        with open(file_path) as file:
            jsonstr = file.read()
        return json.loads(jsonstr)

    @staticmethod
    def dict_to_json(file_path: str, param_dict: dict):
        with open(file_path, "w") as file:
            json.dump(param_dict, file, indent="\t")

    def to_json(self, file_path):
        Handler.dict_to_json(file_path, self.to_dict())

    def hot_reload(self):
        """To call when some modifications are made to the current parts of the model."""
        state = self.to_dict()
        self.from_dict(state, True)

    def from_dict(self, handler_dict, purge=False) -> None:
        """
        Loads a configuration from a dictionnary specifying the perturbations
        and representation of the entire network, modules or tensors\n
        Keys for modules have to be contained in net.named_modules() to be found\n
        Keys for tensors have to be contained in net.named_parameters() to be found\n
        An example of a dictionnary can be found in the file ./profiles/default.json
        """
        self.clusters.change_nb_clusters(handler_dict["nb_clusters"])

        # Represented tensors concat
        tensors_list = handler_dict["tensors"]
        loaded = {ten["name"]: construct_type(self.net, ten) for ten in tensors_list}
        if not purge:
            self.represented_ten |= loaded
        else:
            self.represented_ten = loaded

        # Cluster assignement
        self.assign_clusters()  # TODO read and pass arg `clustering_criterion`

    def to_dict(self) -> dict:
        handler_dict = {
            "nb_clusters": self.clusters.nb_clusters,
            "tensors": [tensor.to_json() for _, tensor in self.represented_ten.items()],
        }
        return handler_dict

    def assign_clusters(self, clustering_criterion: str = "BernoulliXORPerturbation"):
        r"""
        Applies k-means clustering to the perturbation rates of all
        perturbations to group them in the handler's clusters.
        Currently only supports Bitwise Perturbations
        """
        filtered_perts = [
            ten.pert[clustering_criterion]
            for ten in self.represented_ten.values()
            if clustering_criterion in ten.pert
        ]
        assert (
            len(filtered_perts) > 0
        ), f"Trying to cluster with {clustering_criterion} yield 0 represented_tensor"

        # Delegate perts to `Cluster` obj
        self.clusters.assign_perts(filtered_perts)
        self.clusters.cluster()

    def train(self) -> None:  # pragma: no-cover
        self.net.train()

    def eval(self) -> None:  # pragma: no-cover
        self.net.eval()

    def energy_consumption(self) -> Tuple[int, float]:
        r"""Return (max_consumption, current_consumption)
        TODO filter per tensor category ? e.g. parameters/activation have different sums
        """
        energy = [t.energy_consumption() for t in self.represented_ten.values()]
        return sum([t[0] for t in energy]), sum([t[1] for t in energy])

    def change_model_output(
        self, target_output_size: int, freeze_features=True
    ) -> None:
        """Change the model output size to `target_output_size`

        Args:
            target_output_size (int): [description]
        """
        self.net = change_model_output(self.net, target_output_size, freeze_features)
        self.hot_reload()
        for key in OUTPUT_SIZE_ALIAS:
            if key in self.net._hyperparameters:
                self.net._hyperparameters[key] = target_output_size
