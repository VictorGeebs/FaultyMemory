import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import random
import perturbator as P



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

def hook_print_bwd(model, grad_out, grad_in):
    print('')
    print('')
    print(model)
    #print(list(model.named_parameters()))
    print("------------Bwd------------")
    print("Output Gradient (this layer's error)")
    print(grad_out)
    print("Input Gradient (next layer's error)")
    print(grad_in)

def hook_mod_fwd(model, inp, out):
    return torch.Tensor([1., 3.])

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,3)
        self.fc3 = nn.Linear(3,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def hook_all_fwd(model, hook_fn):
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)

def hook_all_bwd(model, hook_fn):
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_backward_hook(hook_fn)

net = SimpleNet()
hooker = P.HookPert()
# Applying hooks to Network

# net.fc1.register_forward_hook(hook_print_fwd)
# net.fc1.register_backward_hook(hook_print_bwd)
# net.fc2.register_forward_hook(hook_print_fwd)
# net.fc2.register_backward_hook(hook_print_bwd)
# net.fc3.register_forward_hook(hook_print_fwd)
# net.fc3.register_backward_hook(hook_print_bwd)
# net.fc1.register_forward_hook(hooker)
hook_all_fwd(net, hook_print_fwd)
hook_all_bwd(net, hook_print_bwd)
hook_all_fwd(net, hooker.perturb)


# TODO: REMOVE
# TESTING FLUFF
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
# Training
loss_y = []
nb_epoch = 1
for epoch in range(nb_epoch):
    #print(list(net.parameters()))

    running_loss = 0.0

    inputs = torch.tensor([0.2, 0.1])
    labels = torch.tensor([1., 0.5])
    
    optimizer.zero_grad()
    #input("Fwd")
    outputs = net.forward(inputs)
    loss = criterion(outputs, labels)
    #input("Bwd")
    loss.backward()
    #input("Step")
    optimizer.step()

    running_loss += loss.item()
    loss_y.append(running_loss)
    print('[%d] loss: %.3f' %
            (epoch + 1, running_loss))
    running_loss = 0.0

