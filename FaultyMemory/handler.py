import json

from FaultyMemory.cluster import Cluster
from FaultyMemory.perturbator import Perturbator
from FaultyMemory.representation import Representation
from FaultyMemory.represented_tensor import RepresentedParameter, RepresentedActivation, construct_type
from FaultyMemory.utils import ten_exists

from typing import Tuple, Union, Optional, Dict


class Handler(object):
    r"""
    Class in charge of saving tensors, storing information about them,
    activation perturbations and clusters.
    """

    def __init__(self, net, clusters=None):
        self.net = net
        self.represented_ten = []

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
        [represented_ten.quantize_perturb() for represented_ten in self.represented_ten]

    def restore(self):
        [represented_ten.restore() for represented_ten in self.represented_ten]

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
        [self.add_tensor(param_key, representation, perturb) for param_key, _ in self.net.named_parameters()]

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
        tensors_dict = handler_dict['weights']['tensors']
        self.represented_ten += [construct_type(ten) for ten in tensors_dict]

        # Cluster assignement
        self.assign_clusters() #TODO read and pass arg `clustering_criterion`

    def to_dict(self):
        handler_dict = {
            "nb_clusters": self.clusters.nb_clusters,
            "tensors": [tensor.to_json() for tensor in self.represented_ten]
        }
        return handler_dict

    def assign_clusters(self, clustering_criterion: str = 'BernoulliXORPerturbation'):
        r"""
        Applies k-means clustering to the perturbation rates of all
        perturbations to group them in the handler's clusters.
        Currently only supports Bitwise Perturbations
        """
        filtered_perts = [ten.pert[clustering_criterion] for ten in self.represented_ten if clustering_criterion in ten.pert]
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