import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
import copy
import math
import numpy as np
import perturbator as P
import cluster as C
import wrapper as W
import handler as H

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,3)
        self.fc3 = nn.Linear(3,4)
        self.fc4 = nn.Linear(4,5)
        self.fc5 = nn.Linear(5,6)
        self.fc6 = nn.Linear(6,7)
        self.fc7 = nn.Linear(7,1)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x



#Defining Networks
net_clean = torchvision.models.resnet18(pretrained=True)
net_dirty = W.Wrap(net_clean)


#Init the handler and clusters
modules = list(net_dirty.children())
pert1 = P.Gauss(p=0.5, mu=0, sigma=1)
pert2 = P.Zeros(p=0.1)
c1 = C.Cluster([pert1])
c2 = C.Cluster([pert2])
clusters = [c1, c2]
handler = H.Handler(net_dirty, clusters)
handler.init_clusters()

