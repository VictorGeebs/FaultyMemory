import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
import torch.optim as optim
import copy
import math
import numpy as np
import perturbator as P
import cluster as C
import handler as H
#import hooks as hooks

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,3)
        self.fc3 = nn.Linear(3,4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.perturb = 1

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Defining Networks
net = SimpleNet()

#Init the handler and clusters
modules = list(net.children())
print(modules)
named_params = net.named_parameters()
params = list(net.parameters())

pert1 = P.Zeros(1)
pert2 = P.Ones(0.5)
c1 = C.TensorCluster(perturb=[pert1])
c2 = C.TensorCluster(perturb=[pert2])
clusters = [c1, c2]
handler = H.Handler(net, clusters)

handler.init_clusters()

inp = torch.tensor([1.])
print(inp)
out = handler.forward(inp)
print(out)
print(handler)
handler.restore_modules()
print(handler)