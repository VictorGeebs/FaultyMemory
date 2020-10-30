import copy
import json
from scipy.cluster.vq import kmeans, vq
from tqdm import tqdm

import FaultyMemory.perturbator as P
import FaultyMemory.representation as R
import FaultyMemory.cluster as C
from FaultyMemory.represented_tensor import RepresentedWeight, RepresentedActivation
from FaultyMemory.hook import *
from FaultyMemory.utils import listify, ten_exists

from typing import List, Tuple, Union, Optional


class Handler():
    r"""
    Class in charge of saving tensors, storing information about them,
    activation perturbations and clusters.
    """

    def __init__(self, net, clusters=None):
        self.net = net
        self.saved_net = copy.deepcopy(net)
        self.clusters = clusters if clusters is not None else []
        self.modules = list(self.net.children())
        self.tensor_info = {}
        self.acti_info = {}
        self.hooks = {}
        self.clustering = False

    def __str__(self):
        print("Handler: ")
        for name in self.tensor_info:
            print(name, ": ", self.tensor_info[name])
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
        self.save_net()
        self.perturb_tensors()
        out = self.net.forward(x)
        self.restore()
        return out

    def restore(self):
        r"""
        Copies the modules' saved parameters back to the normal weights to allow for backpropagation
        """
        pert_params = list(self.net.parameters())
        saved_params = list(self.saved_net.parameters())
        for perturbed, saved in zip(pert_params, saved_params):
            # perturbed_shape = perturbed.shape
            assert perturbed.shape == saved.shape, 'Size mismatch, there is an error somewhere'
            # perturbed = perturbed.flatten()
            # saved = saved.flatten()
            # perturbed.data = saved.data
            # perturbed = perturbed.view(perturbed_shape)
            # saved = saved.view(perturbed_shape)
            perturbed.data.copy_(saved.data) 

    def add_tensor(self, 
                   name: str, 
                   perturb: Union[List, Perturbator] = None,
                   representation: Representation = None,
                   reference: Optional[str] = None):
        if reference is None:
            ten_exists(self.net.named_parameters(), name)
            reference = self.net.named_parameters()[name]
        perturb = listify(perturb)  # ensure perturb is a list
        self.tensor_info[name] = RepresentedWeight(reference, perturb, representation)

    def remove_tensor(self, name):
        self.tensor_info.pop(name, None)

    def add_module(self, name, perturb=None, representation=None):
        net_dict = self.net.named_modules()
        ten_exists(net_dict, name)
        module = self.net.named_modules()[name]
        for param_key, _ in module.named_parameters():
            full_key = name + '.' + param_key
            self.add_tensor(full_key, perturb, representation)

    def remove_module(self, name):
        module = self.net.named_modules()[name]
        for param_key, _ in module.named_parameters():
            full_key = name + '.' + param_key
            self.remove_tensor(full_key)

    def add_all_tensors(self, perturb=[None], representation=None):
        for param_key, _ in self.net.named_parameters():
            self.add_tensor(param_key, perturb, representation)

    def remove_all_tensors(self):
        for param_key, _ in self.net.named_parameters():
            self.remove_tensor(param_key)

    def add_activation(self, name, perturb=None, representation=None):
        net_dict = self.net.named_modules()
        ten_exists(net_dict, name)
        module = net_dict[name]
        hook = Hook(perturb, representation)
        self.hooks[name] = module.register_forward_hook(hook.hook_fn)
        self.acti_info[name] = (perturb, representation)

    def remove_activation(self, name):
        self.hooks[name].remove()
        self.acti_info.pop(name, None)

    def add_network_activations(self, perturb=None, representation=None):
        for module, _ in self.net.named_modules():
            self.add_activation(module)

    def clear_network_activations(self):
        for module, _ in self.net.named_modules():
            self.remove_activation(module)

    def save_net(self):
        """
        Creates a backup of the network in saved_net
        """
        self.saved_net = copy.deepcopy(self.net)

    def perturb_tensors(self, scaling: str = 'none'):
        """
        Applies every perturbation specified in their tensor_info to each
        tensor.\n
        Tensors are modified in-place, without modifying their reference.
        """
        if self.clustering is True:
            for cluster in self.clusters:
                cluster.perturb_tensors()
        else:
            for _, (tens, pert, repr) in tqdm(self.tensor_info.items(), 'Perturbing tensors'):
                if repr is not None:
                    repr.convert_tensor(tens)
                if pert is not None:
                    for perturb in pert:
                        if perturb is not None:
                            perturb(tens, repr, scaling)

    def from_json(self, file_path):
        """
        Creates a handler dictionnary from a json file and initializes the handler to that configuration
        """
        with open(file_path) as file:
            jsonstr = file.read()
            handler_dict = json.loads(jsonstr)
            self.from_dict(handler_dict)

    def from_dict(self, handler_dict):
        """
        Loads a configuration from a dictionnary specifying the perturbations
        and representation of the entire network, modules or tensors\n
        Keys for modules have to be contained in net.named_modules() to be found\n
        Keys for tensors have to be contained in net.named_parameters() to be found\n
        An example of a dictionnary can be found in the file ./profiles/default.json
        """
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
                if pert_list is not None:
                    for pert_dict in pert_list:
                        pert = P.construct_pert(pert_dict)
                        perturbs.append(pert)

                for (tensor_name, ref) in list(self.net.named_parameters()):
                    self.tensor_info[tensor_name] = (ref, perturbs, repr)

            # Modules batch
            modules = weight_dict['modules']
            if modules is not None:
                for module in modules:
                    module_name = module['name']

                    repr_dict = module['repr']
                    repr = R.construct_repr(repr_dict)

                    pert_list = module['perturb']
                    perturbs = []
                    if pert_list is not None:
                        for pert_dict in pert_list:
                            pert = P.construct_pert(pert_dict)
                            perturbs.append(pert)
                    else:
                        perturbs=None
                    
                    current_mod = self.net.named_modules()[module_name]
                    for param_key, _ in current_mod.named_parameters():
                        full_key = module_name + '.' + param_key
                        tens = current_mod.named_parameters()[param_key]
                        self.tensor_info[full_key] = (tens, perturbs, repr)

            # Tensors
            tensors = weight_dict['tensors']
            if tensors is not None:
                for tensor in tensors:
                    tensor_name = tensor['name']

                    repr_dict = tensor['repr']
                    repr = R.construct_repr(repr_dict)

                    pert_list = tensor['perturb']
                    perturbs = []
                    if pert_list is not None:
                        for pert_dict in pert_list:
                            pert = P.construct_pert(pert_dict)
                            perturbs.append(pert)
                    else:
                        perturbs=None

                    tens = self.net.named_parameters()[tensor_name]
                    self.tensor_info[tensor_name] = (tens, perturbs, repr)

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
                    self.hooks[name] = module.register_forward_hook(
                        hook.hook_fn)
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

                    current_mod = self.net.named_modules()[module_name]
                    hook = Hook(perturbs, repr)
                    self.hooks[module_name] = current_mod.register_forward_hook(
                        hook.hook_fn)
                    self.acti_info[module_name] = (perturbs, repr)

        # Cluster assignement
        self.assign_clusters()

    def to_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent="\t")

    def to_dict(self):
        handler_dict = {
            "nb_clusters": len(self.clusters)
        }

        # Tensors
        tensor_list = []
        for name in self.tensor_info:
            tensor_list.append(self.tensor_to_json(name))

        handler_dict["weights"] = {
            "net": None,
            "modules": None,
            "tensors": tensor_list
        }

        # Activations
        acti_list = []
        for name in self.acti_info:
            acti_list.append(self.acti_to_json(name))

        handler_dict["activations"] = {
            "net": None,
            "modules": acti_list
        }
        return handler_dict

    def tensor_to_json(self, tensor_name):
        """
        Creates a dict representing the tensor information to be later
        converted into json format
        """
        name = tensor_name
        (_, pert_list,repr) = self.tensor_info[name]

        repr_data = repr.to_json() if repr is not None else None
        pert_data = [o.to_json()
                     for o in pert_list] if pert_list is not None else None

        tensor_dict = {
            "name": name,
            "repr": repr_data,
            "perturb": pert_data
        }

        return tensor_dict

    def acti_to_json(self, tensor_name):
        """
        Creates a dict representing the activation information to be later
        converted into json format
        """
        name = tensor_name
        pert_list, repr = self.acti_info[name]

        repr_data = repr.to_json() if repr is not None else None
        pert_data = [o.to_json()
                     for o in pert_list] if pert_list is not None else None

        acti_dict = {
            "name": name,
            "repr": repr_data,
            "perturb": pert_data
        }

        return acti_dict

    def assign_clusters(self):
        """
        Applies k-means clustering to the perturbation rates of all
        perturbations to group them in the handler's clusters.
        Currently only supports Bitwise Perturbations
        """
        running_perts = {}
        for name in self.tensor_info:
            item = self.tensor_info[name]
            pert_list = item[1]
            pert_names = []
            prob_list = []
            if pert_list is not None:
                for pert in pert_list:
                    pert_names.append(pert.__class__.__name__)
                    prob_list.append(pert.p)
            pert_names = '_'.join(pert_names)
            if pert_names not in running_perts:
                running_perts[pert_names] = [(name, prob_list)]
            else:
                running_perts[pert_names].append((name, prob_list))

        running_perts.pop('')

        assert len(running_perts) <= len(self.clusters), "More different perturbations than clusters available, cannot assign tensors to clusters"

        # ONLY BITWISEPERT FOR THE TIME BEING
        bitwises = running_perts['BitwisePert']
        bitwise_probs = [item[1][0] for item in bitwises]
        centers, _ = kmeans(bitwise_probs, len(self.clusters))
        groups, _ = vq(bitwise_probs, centers)

        for tensor, cluster in zip(bitwises, groups):
            name = tensor[0]
            tensor_ref = self.tensor_info[name][0]
            repr = self.tensor_info[name][2]
            self.clusters[cluster].add_tensor(tensor_ref, repr)

        for cluster, rate in zip(self.clusters, centers):
            pert_dict = {
                "name": "BitwisePert",
                "p": rate}
            pert = P.construct_pert(pert_dict)
            cluster.set_perturb([pert])

    def toggle_clustering(self):
        """
        Turns on or off clustering, which groups tensor perturbations with
        nearby perturbation rates.
        """
        self.clustering = not self.clustering
        return self.clustering
                        
    def train(self) -> None:
        self.net.train()
        
    def eval(self) -> None:
        self.net.eval()

    def get_all(self) -> dict: #TODO en quête d'un meilleur nom
        return dict(self.tensor_info, **self.acti_info) 

    def energy_consumption(self) -> Tuple(int, float):
        r""" Return (max_consumption, current_consumption)
        """
        energy = [t.energy_consumption() for t in self.get_all()]
        return sum([t[0] for t in energy]), sum([t[1] for t in energy])