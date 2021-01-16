"""
Introduction
------------
The FaultyMemory library is a library designed to allow you to simulate memory 
faults anywhere in your network or module.



"""
import torch
import torchvision
import torchvision.transforms as transforms
import Dropit as Dropit  # These imports define the structure of the network we will be using
import wrn_mcdonnell_manual as McDo 


# Importing network, regular pytorch
net_path = './models/mcdonnell.pth'

net = McDo.WRN_McDonnell(depth=28, width=10, num_classes=10, dropit=False, actprec=3)  # Defining the network we will use

state_dict = torch.load(net_path, map_location=torch.device('cpu'))['model_state_dict']  # Importing the trained network weights
net.load_state_dict(state_dict)


# Creating a testloader to test our network, regular pytorch
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048,
                                         shuffle=False,
                                         num_workers=2)


"""
Once you have a network you are ready to work with, you must create a Handler
instance that will deal with your model. It is the handler that will take care
of saving the types of perturbation and the representation you wish to use in
your model. To do so, simply instanciate a Handler object, passing your network
as a constructor parameter.
"""
import FaultyMemory.handler as H

my_handler = H.Handler(net)

"""
Next, to configure which parameters (individual tensors, sub-modules like layers
and blocks or the entire network) you wish to perturb, there are two possible
ways to do so.
1 - Add them manually to the handler with handler.add_tensor(), add_module() or
    add_network()
2 - Create or modify a JSON configuration file like the ones found in ./profiles

Let's start with method one: Adding parameters manually
parameters you wish to perturb should be in the network named_parameters so the
handler can have easy access to and so that configuration may be saved later.
However, it is still possible to pass a tensor by reference if you wish to do so.

Let's say we have a simple network and its entirety is using the same
representation and will be under the same type of perturbation.
We will start by creating a perturbation instance and a representation instance.
"""
import FaultyMemory.perturbator as P
import FaultyMemory.representation as R

zero_pert = P.Zeros(p=0.1)  # A stuck-at-zero perturbation with a probability of p
int5_repr = R.Representation(width=5, unsigned=False)  # A 5 bit wide signed integer representation

my_handler.add_network([zero_pert], int5_repr)  # When adding perturbations, lists are used because you might want to have a series of different perturbations on the same tensors

"""
This network is now ready to be used and tested. Just like a regular module,
we can calculate an output by calling handler.forward(x) or simply handler(x)
with x as our input tensor.
Doing so will apply the specified perturbations to the network and calculate
the output in the specified representation. To restore the network to its
original parameter values, we can call handler.restore(). This should be done
before another forward pass is done, else the errors past will stack with the
new ones.
A function exists to test a perturbed network's accuracy in our utils,
test_accuracy(), given you provide it with a handler and a testloader.
This step can take a while depending on the type of perturbation and the size
of the network.
"""


"""
Let's say we wish to leave the second to last layer of our network clean and
undisturbed, and change the last layer's weight to be binary and without
perturbation.
We can do so with the remove_module() or remove_tensor() functions, and the
add_module() or add_tensor() as the new parameters given will override the old
ones. Our last two modules happen to be called conv_last and bn_last.
"""

my_handler.remove_module("conv_last")
binary_repr = R.BinaryRepresentation(unsigned=False)
my_handler.add_tensor("bn_last.weight", None, binary_repr)

"""
If we wish to perturb the activations of certain modules in the network, we can
do so with the add_activation() and add_network_activation() functions in the
same matter as the other functions.
"""

my_handler.add_activation("conv_last", [zero_pert], int5_repr)

"""
We can save this configuration for future use with the handler.to_json()
function.
"""

my_handler.to_json('./profiles/saved_handler.json')


"""
Now for method two: Loading a saved profile or creating your own
The JSON format is used to store handler data, and a profile should look like
this:

{
    "nb_clusters": 1,
    "weights":
    {
        "net":
        {
            "repr":
            {
                "name": "BinaryRepresentation",
                "unsigned": false,
                "width":1
            },
            "perturb":[
                {
                    "name":"BitwisePert",
                    "p":0.2
                }
            ]
        },
        "modules": [
            {
                "name":"conv_last",
                "repr": null,
                "perturb":[null]
            },
            {
                "name":"bn_last",
                "repr": null,
                "perturb":[null]
            }
        ],
        "tensors": [
            {
                "name":"filter1",
                "repr":null,
                "perturb":[
                    {
                        "name":"Zeros",
                        "p":0.5
                    }
                ]
            }
        ]
    },
    "activations":
    {
        "net": null,
        "modules":null
    }
}

To unwrap this, let's go step by step. The "weights" section dictates which
parameters (weights and biases) will be perturbed. You can specify an entire
network under the "net" field, specific modules or layers in the "modules" list
and specific parameters in the "tensors" list. These modules and parameters
need to be in the net.named_modules() and net.named_parameters() respectively
in order to be found by the constructor.
Specifying representations and perturbations is done by giving the name of the
desired class in the "name" field, and creating a field for each constructor
argument.
If no representation or perturbation is wanted, specifying null
respectively will void that field.
The same principles hold true for activations.
Note: the saved file from handler.to_json() is not guaranteed to match the one
used to load it, as there are multiple inputs that can give the same output,
and for the time being we adress everything by tensor instead of looking to
group tensors to simplify the saved file.
"""

import FaultyMemory.utils as utils
accuracy = utils.test_accuracy(my_handler, testloader)

config_path = './profiles/saved_handler.json'
my_handler.from_json(config_path)
