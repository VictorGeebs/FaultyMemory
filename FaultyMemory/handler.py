import json

from FaultyMemory.cluster import Cluster
from FaultyMemory.perturbator import Perturbator
from FaultyMemory.representation import Representation, FixedPointRepresentation
from FaultyMemory.represented_tensor import RepresentedParameter, RepresentedActivation, construct_type
from FaultyMemory.utils import ten_exists
import copy

from typing import Tuple, Union, Optional, Dict


class Handler(object):
    r"""
    Class in charge of saving tensors, storing information about them,
    activation perturbations and clusters.
    """

    def __init__(self, net, clusters=None):
        self.net = net
        self.represented_ten = {}

        self.clusters = Cluster()

    def __str__(self):
        print("Handler: ")
        for ten in self.represented_ten:
            print(ten)
        return ''

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        r"""
        Saves every tensor, then perturbs every tensor, and then makes the
        forward pass

        Args:
            x: Input to process
        """
        self.perturb_tensors()
        out = self.net.forward(x)
        self.restore()
        return out

    def perturb_tensors(self):
        [represented_ten.quantize_perturb() for _, represented_ten in self.represented_ten.items()]

    def restore(self):
        [represented_ten.restore() for _, represented_ten in self.represented_ten.items()]

    def compute_MSE(self):
        return [represented_ten.quantize_MSE() for _, represented_ten in self.represented_ten.items()]

    def value_range(self):
        return [represented_ten.value_range() for _, represented_ten in self.represented_ten.items()]

    def assign_representation_range(self):
        old_prec_list = []
        old_width_list = []
        old_whole_list = []
        
        prec_list = []
        width_list = []
        whole_list = []
        for name, represented_ten in self.represented_ten.items():
            
            old_prec_list.append(represented_ten.repr.nb_digits)
            old_width_list.append(represented_ten.repr.width)
            old_whole_list.append(represented_ten.repr.width - represented_ten.repr.nb_digits)
            
            whole = represented_ten.compute_precision()

            assert isinstance(represented_ten.repr, FixedPointRepresentation)
            precision = represented_ten.repr.width - whole
            
            assert precision >= 0

            prec_list.append(precision)
            width_list.append(represented_ten.repr.width)
            whole_list.append(whole)

            represented_ten.repr.nb_digits = precision
            
        print("old_prec: ", old_prec_list)
        print("old_width: ", old_width_list)
        print("old_whole: ", old_whole_list)
        print(prec_list)
        print(width_list)
        print(whole_list)

    def add_parameter(self, 
                   name: str, 
                   representation: Representation,
                   perturb: Optional[Union[Dict, Perturbator]] = None):
        assert name not in self.represented_ten
        self.represented_ten[name] = RepresentedParameter(self.net, name, representation, perturb)

    def add_activation(self, 
                   name: str, 
                   representation: Representation,
                   perturb: Optional[Union[Dict, Perturbator]] = None):
        assert name not in self.represented_ten
        self.represented_ten[name] = RepresentedActivation(self.net, name, representation, perturb)

    def remove_tensor(self, name):
        self.represented_ten.pop(name, None)

    def add_module_parameters(self, 
                   name: str, 
                   representation: Representation,
                   perturb: Optional[Union[Dict, Perturbator]] = None):
        net_dict = self.net.named_modules()
        ten_exists(net_dict, name)
        module = net_dict[name]
        [self.add_tensor(f'{name}.{param_key}', representation, perturb) for param_key, _ in module.named_parameters()]

    def remove_module_parameters(self, name: str):
        module = self.net.named_modules()[name]
        [self.remove_tensor(f'{name}.{param_key}') for param_key, _ in module.named_parameters()]

    def add_net_parameters(self, 
                           representation: Representation, 
                           perturb: Optional[Union[Dict, Perturbator]] = None):
        [self.add_parameter(param_key, representation, perturb) for param_key, _ in self.net.named_parameters()]

    def remove_net_parameters(self):
        [self.remove_tensor(param_key) for param_key, _ in self.net.named_parameters()]

    def add_net_activations(self, 
                            representation: Representation, 
                            perturb: Optional[Union[Dict, Perturbator]] = None):
        [self.add_activation(module, representation, perturb) for module, _ in self.net.named_modules()]

    def remove_net_activations(self):
        [self.remove_activation(module) for module, _ in self.net.named_modules()]

    def from_json(self, file_path):
        """
        Creates a handler dictionnary from a json file and initializes the handler to that configuration
        """
        with open(file_path) as file:
            jsonstr = file.read()
            handler_dict = json.loads(jsonstr)
            self.from_dict(handler_dict)

    def to_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent="\t")

    def from_dict(self, handler_dict):
        """
        Loads a configuration from a dictionnary specifying the perturbations
        and representation of the entire network, modules or tensors\n
        Keys for modules have to be contained in net.named_modules() to be found\n
        Keys for tensors have to be contained in net.named_parameters() to be found\n
        An example of a dictionnary can be found in the file ./profiles/default.json
        """
        self.clusters.change_nb_clusters(handler_dict['nb_clusters'])

        # Represented tensors concat
        tensors_list = handler_dict['tensors']
        
        self.represented_ten = {**self.represented_ten, **{ten["name"]: construct_type(self.net, ten) for ten in tensors_list}}

        # Cluster assignement
        self.assign_clusters() #TODO read and pass arg `clustering_criterion`

    def to_dict(self):
        handler_dict = {
            "nb_clusters": self.clusters.nb_clusters,
            "tensors": [tensor.to_json() for _, tensor in self.represented_ten.items()]
        }
        return handler_dict

    def assign_clusters(self, clustering_criterion: str = 'BernoulliXORPerturbation'):
        r"""
        Applies k-means clustering to the perturbation rates of all
        perturbations to group them in the handler's clusters.
        Currently only supports Bitwise Perturbations
        """
        filtered_perts = [ten.pert[clustering_criterion] for ten in self.represented_ten.values() if clustering_criterion in ten.pert]
        assert len(filtered_perts) > 0, f'Trying to cluster with {clustering_criterion} yield 0 represented_tensor'

        # Delegate perts to `Cluster` obj
        self.clusters.assign_perts(filtered_perts)
        self.clusters.cluster()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def energy_consumption(self) -> Tuple[int, float]:
        r""" Return (max_consumption, current_consumption)
        TODO filter per tensor category ? e.g. parameters/activation have different sums
        """
        energy = [t.energy_consumption() for t in self.represented_ten]
        return sum([t[0] for t in energy]), sum([t[1] for t in energy])