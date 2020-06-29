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
import wrapper as W
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
#net_clean = torchvision.models.resnet18(pretrained=True)
#net_dirty = W.Wrap(net_clean)
net = SimpleConv()

#Init the handler and clusters
modules = list(net.children())
print(modules)
named_params = net.named_parameters()
#print("Params:\n", list(named_params))
params = list(net.parameters())
#print("Params:\n", params)
# print("params[0] shape:\n", list(params)[0].shape)
# print("params[0]:\n", list(named_params)[0])
# print("params[0][0]:\n", list(params)[0][0])
# print("params[0][0] shape:\n", list(params)[0][0].shape)
filter1 = params[0]
filter2 = params[0]
filter3 = params[0]
conv2 = modules[2]
#print(filter1)
#print(filter2)
#print(filter3)
#print(conv2)

#print(list(conv2.parameters())[0].shape)


pert1 = P.Gauss(p=1, mu=0, sigma=1)
pert2 = P.Zeros(p=1)
# hookpert1 = P.HookPert()
c1 = C.Cluster([pert1])
c2 = C.Cluster([pert2])
clusters = [c1, c2]
handler = H.Handler(net, clusters)
# handler.init_clusters()

#c1.add_tensor(filter1)
#c1.add_tensor(filter3)
c2.add_tensor(filter2)
c2.add_model(conv2)


#c1.perturb_models()
#c2.perturb_models()

print("\nAfter\n")
#print(filter1)
#print(filter2)
#print(filter3)
#print(list(modules[2].parameters()))

#print("C1: \n", c1)
#print("C2: \n", c2)

c2.remove_model(conv2)
#print("C2: \n", c2)
#c2.remove_tensor(filter2)

#print("Params After:\n", list(net.named_parameters()))


# def hook_all_fwd(model, hook_fn):
#     hooks = {}
#     for name, module in model.named_modules():
#         hooks[name] = module.register_forward_hook(hook_fn)

# def hook_all_bwd(model, hook_fn):
#     hooks = {}
#     for name, module in model.named_modules():
#         hooks[name] = module.register_backward_hook(hook_fn)

# def hook_print_fwd(model, inp, out):
#     print('')
#     print('')
#     print(model)
#     #print(list(model.named_parameters()))
#     print("------------Fwd------------")
#     print("Input Activations")
#     print(inp)
#     print("Calculated Output")
#     print(out)

# hook_all_fwd(net, hook_print_fwd)

# for cluster in clusters:
#     cluster.apply_hooks()




# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

# inputs = torch.tensor([0.1])
# labels = torch.tensor([1.])

# optimizer.zero_grad()
# outputs = net.forward(inputs)
# loss = criterion(outputs, labels)
# loss.backward()
# optimizer.step()