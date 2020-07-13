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
import utils


PATH = './models/xor_net.pth'


testset = utils.R2Dataset(2, 10000)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Loading Net
net = utils.Xor()
net.load_state_dict(torch.load(PATH))


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
