import os
import sys
import torch
import torchvision
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
import time


class R2Dataset(Dataset):
    """
    A dataset for testing purposes.
    This dataset contains values of R^2 with labels as the XOR of the sign of those two values.
    """
    def __init__(self, dim=1, len=1):
        self.dim = dim
        self.len = len
        self.samples = []
        self.generate_set()

    def generate_set(self):
        for i in range(self.len):
            data = self.gen_sample(self.dim)
            label = self.label_sample(data)
            self.samples.append([data, label])

    def gen_sample(self, dim=1, mu=0, sigma=1):
        sample = torch.zeros([dim])
        for x in range(dim):
            sample[x] = random.gauss(mu, sigma)
        return sample

    def label_sample(self, sample):
        poscount=0
        for value in sample:
            poscount += (value >= 0)
        return (poscount%2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def test_accuracy(net, testloader) ->float:
    """
    A basic test function to test the accuracy of a network. \n
    This function might need modification depending on the type of label you wish to have.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            samples, labels = data
            outputs = net(samples)
            #net.restore()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("running acc: ", correct/total)
            # break
    accuracy = correct/total
    return accuracy


def train_net(net, optimizer, criterion, trainloader, nb_epochs, prt=True):
    """
    A basic function used to train networks in a generic fashion
    """
    for epoch in range(nb_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if prt==True:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/(i+1)))
    if prt==True:
        print('Finished Training')

def generate_graphs(net, testloader, probs): # DEPRECATED
    """
    This function is deprecated because of the use of generate_point
    This function was originally designed to simulate datapoints for an entire networked perturbed the same way
    """
    clean_accuracy = []
    pert_accuracy = []
    acti_accuracy = []
    both_accuracy = []
    
    for prob in probs:
        clean, pert, acti, both = generate_point(net, testloader, prob)
        clean_accuracy.append(clean)
        pert_accuracy.append(pert)
        acti_accuracy.append(acti)
        both_accuracy.append(both)
        print("prob %3.5f done" %prob)
    
    return clean_accuracy, pert_accuracy, acti_accuracy, both_accuracy

def generate_point(net, testloader, prob): # DEPRECATED
    """
    This function is deprecated because of the use of clusters
    Handlers now store tensor information themselves
    """
    net_pert = copy.deepcopy(net)
    pert_pert = P.Zeros(prob)
    c_pert = C.Cluster([pert_pert], networks=[net_pert])
    handler_pert = H.Handler(net_pert, [c_pert])

    net_acti = copy.deepcopy(net)
    pert_acti = P.Zeros(prob)
    c_acti = C.Cluster([pert_acti], network_activations=[net_acti])
    handler_acti = H.Handler(net_acti, [c_acti])

    net_both = copy.deepcopy(net)
    pert_both = P.Zeros(prob)
    c_both = C.Cluster([pert_both], networks=[net_both], network_activations=[net_both])
    handler_both = H.Handler(net_both, [c_both])

    clean_accuracy = test_accuracy(net, testloader)
    pert_accuracy = test_accuracy(handler_pert, testloader)
    acti_accuracy = test_accuracy(handler_acti, testloader)
    both_accuracy = test_accuracy(handler_both, testloader)

    return clean_accuracy, pert_accuracy, acti_accuracy, both_accuracy


class Xor(nn.Module):
    """
    A simple network to solve R^2 XOR datasets
    """
    def __init__(self):
        super(Xor, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
