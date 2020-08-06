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
import representation as R
import cluster as C
import handler as H
import random
import json

import wrn_mcdonnell_manual as McDo
import Dropit
from collections import OrderedDict

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

#net = McDo.WRN_McDonnell(depth=28, width=10, num_classes=10, dropit=False, actprec=3)
net = SimpleNet()
handler = H.Handler(net)

hook_all_fwd(net, hook_print_fwd)
#param_list = list(net.named_parameters())
#print(param_list[0][0])
# mod_dict = dict(net.named_modules())
# mod = mod_dict['fc1']
# print(mod)
# param_dict = dict(mod.named_parameters())
# #print(param_dict)
# for key in param_dict:
#     print(key)
#     print(param_dict[key])
#pert = P.Zeros(p=1)

with open('./profiles/default.txt') as file:
    jsonstr = file.read()
    handlerDict = json.loads(jsonstr)
    handler.from_json(handlerDict)
    #for el in handler.net.modules():
    #    el.register_forward_hook(pert.hook)
    #print(data['tensors'][0]['repr']['unsigned'])

print(handler.tensor_info)
print(handler.acti_info)
print(handler.hooks)

inp = torch.tensor([1.])
out = handler(inp)
print('out: ', out)







"""
def confint_mean_testset():
  top1avgs = []
  while True:
    top1avgs.append(test(model, loss_func, args.device, args.testloader, len(top1avgs), wm))
    confint = sms.DescrStatsW(top1avgs).tconfint_mean()
    if (confint[1]-confint[0] < args.confidence_interval_test and len(top1avgs) >5):
        break
"""