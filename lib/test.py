import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import random
import copy
import numpy as np
import perturbator as P
import cluster as C
import handler as H
import utils
import wrn_mcdonnell_manual as McDo
import Dropit
from collections import OrderedDict
import time

PATH = './models/mcdonnell.pth'
XOR = './models/xor_net.pth'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

xor_set = utils.R2Dataset(2, 10000)
xor_loader = torch.utils.data.DataLoader(xor_set, batch_size=1, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False, num_workers=2)

# Loading Net
xor_net = utils.Xor()
xor_net.load_state_dict(torch.load(XOR))


#init params: depth=28, width=10
net = McDo.WRN_McDonnell(depth=28, width=10, num_classes=10, dropit=False, actprec=3)
state_dict = torch.load(PATH, map_location=torch.device('cpu'))['model_state_dict']

net.load_state_dict(state_dict)

#for param in list(net.named_parameters()):
#    print(param[0])
#print(list(net.named_parameters())[0][0])

pert = P.BitwisePert(p=0)
repr_bin = P.BinaryRepresentation()
c1 = C.Cluster([pert], repr=repr_bin)
m1 = list(net.parameters())[0:26]
for param in m1:
    c1.add_tensor(param)


repr6 = P.Representation(width=3, unsigned=False)
c2 = C.Cluster([pert], repr=repr6)
m2 = list(net.parameters())[26:28]
for param in m2:
    c2.add_tensor(param)

handler = H.Handler(net, [c1])



print("starting testing")
start_time = time.time()

results = utils.test_accuracy(handler, testloader)
tot_time = time.time()-start_time
print("Acc: ", results)
print("Time: ", tot_time)


"""

probs = np.logspace(-0.1, -2.5, 20)
print(probs)

clean_accuracy, pert_accuracy, acti_accuracy, both_accuracy = utils.generate_graphs(net, testloader, probs)

plt.figure(1)
plt.plot(probs, clean_accuracy, probs, pert_accuracy, probs, acti_accuracy, probs, both_accuracy)
plt.title("Accuracy with regard to probability")
plt.xlabel("probability")
plt.ylabel("accuracy")
plt.legend(["clean", "only weights", "only activations", "weights and actis"])
plt.show()
"""