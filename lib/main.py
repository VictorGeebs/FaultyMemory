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
import random






def hook_all_fwd(model, hook_fn):
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)

def hook_print_fwd(model, inp, out):
    print('')
    print('')
    print(model)
    #print(list(model.named_parameters()))
    print("------------Fwd------------")
    print("Input Activations")
    print(inp)
    print("Calculated Output")
    print(out)

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


rado = np.random.binomial(1, 0.5, (4, 5, 3))
print(rado)
nums = np.packbits(rado, axis=0, bitorder='little')[0]
print(nums)


"""
#Defining Networks
net = SimpleNet()

#Init the handler and clusters
modules = list(net.children())
print(modules)


named_params = net.named_parameters()
print(list(named_params))
params = list(net.parameters())


pert1 = P.BitwisePert(p=0.1)
pert2 = P.BitwisePert(p=0.1)
c1 = C.Cluster(perturb=[pert1], activations=[modules[0], modules[2]])
c2 = C.Cluster(perturb=[pert2], activations=[modules[1]])
clusters = [c1, c2]
handler = H.Handler(net, clusters)
handler.init_clusters()
handler.perturb_modules()
print(list(net.named_parameters()))
#print(handler)
hook_all_fwd(net, hook_print_fwd)
handler.move_activation(c2, modules[2])



inp = torch.Tensor([1.])
out = handler.forward(inp)
print("")
print(out)


def confint_mean_testset():
  top1avgs = []
  while True:
    top1avgs.append(test(model, loss_func, args.device, args.testloader, len(top1avgs), wm))
    confint = sms.DescrStatsW(top1avgs).tconfint_mean()
    if (confint[1]-confint[0] < args.confidence_interval_test and len(top1avgs) >5):
        break
"""