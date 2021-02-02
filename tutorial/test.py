import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tqdm import tqdm
from BinaryWeightMemory import BinaryWeightMemory
import json
import time
import wrn_mcdonnell_manual as McDo
import FaultyMemory.utils as utils
import FaultyMemory.handler as H
import FaultyMemory.cluster as C
import FaultyMemory.perturbator as P
import numpy as np
import copy
import random
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch

torch.cuda.device(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = './models/mcdonnell.pth'
SCALING = 'none'
SCALINGS = ['none', 'he', 'mean']

# all weights binary

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=2)

# init params: depth=28, width=10
net = McDo.WRN_McDonnell(
    depth=28, width=10, num_classes=10, dropit=False, actprec=3).to(device)
state_dict = torch.load(PATH, map_location=device)[
                        'model_state_dict']
lanmax_state_space = torch.load(PATH, map_location=device)[
                                'lanmax_state_space']
net.load_state_dict(state_dict)
net.device = device
net.eval()

handler = H.Handler(net)
handler.from_json('./profiles/McDo.json')

wm = BinaryWeightMemory(net, p=0.01, scaling=SCALING)
wm.pis = lanmax_state_space


def iter_weights():
    weights = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            y = np.bincount(m.weight.data.numpy().flatten().astype(int) + 2)
            ii = np.nonzero(y)[0]
            print(list(zip(ii, y[ii])))
            weights.append(copy.deepcopy(m.weight.data.numpy()))
    return weights


def inspect():
    handler.perturb_tensors('none')
    weights = iter_weights()
    handler.restore()

    idx = 0
    handler.perturb_tensors('none')
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            res = m.weight.data.numpy().flatten().astype(int) + 2
            y = np.bincount(res)
            ii = np.nonzero(y)[0]
            print((m.weight.data.numpy() == weights[idx]).sum())
            assert((m.weight.data.numpy() == weights[idx]).sum() > 0)
            idx += 1
    handler.restore()

    idx = 0
    handler.from_json('./profiles/McDo_faultfree.json')
    handler.perturb_tensors('none')
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            res = m.weight.data.numpy().flatten().astype(int) + 2
            y = np.bincount(res)
            ii = np.nonzero(y)[0]
            print(list(zip(ii, y[ii])))
            assert((m.weight.data.numpy() == weights[idx]).sum() > 0)
            idx += 1
    handler.restore()


def test_lib(profile: str = './profiles/McDo_faultfree.json', nb_nets: int = 5):
    acc_dict = {}
    handler.from_json(profile)

    for scaling in SCALINGS:
        avg_list = []
        for _ in range(nb_nets):
            handler.perturb_tensors(scaling)
            avg_list.append(utils.test_accuracy(net, testloader))
            handler.restore()
        acc_dict[scaling] = avg_list

    return acc_dict


def test_legacy(profile: np.ndarray = np.zeros_like(lanmax_state_space), nb_nets: int = 5): 
    acc_dict = {}
    wm.pis = profile

    for scaling in SCALINGS:
        avg_list = []
        wm.scaling = scaling
        for _ in range(nb_nets):
            wm.binarize()
            avg_list.append(utils.test_accuracy(net, testloader))
            wm.restore()
        acc_dict[scaling] = avg_list

    return acc_dict


def test_dset_zerop():
    lib_res = test_lib(nb_nets=1)
    leg_res = test_legacy(nb_nets=1)
    print("Zero probability")
    print("LEG Accuracies ", leg_res)
    print("LIB Accuracies ", lib_res)


def test_dset_trainedp():
    torch.manual_seed(0)
    np.random.seed(0)
    lib_res = test_lib(profile='./profiles/McDo.json', nb_nets=5)
    torch.manual_seed(0)
    np.random.seed(0)
    leg_res = test_legacy(profile=lanmax_state_space, nb_nets=5)
    print("Trained probability")
    print("LEG Accuracies ", leg_res)
    print("LIB Accuracies ", lib_res)


def test_egality():
    wm.pis = np.zeros_like(lanmax_state_space)
    start = time.time()
    wm.binarize()
    weights = iter_weights()
    wm.restore()
    print(f'Legacy took {time.time() - start}')

    handler.from_json('./profiles/McDo_faultfree.json')
    start = time.time()
    handler.perturb_tensors(SCALING)
    weights_lib = iter_weights()
    handler.restore()
    print(f'Lib took {time.time() - start}')

    for leg, lib in zip(weights, weights_lib):
        assert((leg != lib).sum() == 0)

# pert_index = 0
# for name in dict(net.named_parameters()):
#     info = handler.tensor_info[name]
#     if info[1] is not None:
#         pert = info[1][0]
#         #pert.set_probability(0.5)
#         pert.set_probability(lanmax_state_space[pert_index])
#     pert_index += 1

# print("perturbing")
# handler.perturb_tensors(scaling=True)
# print("perturbed")
# handler.to_json('./profiles/McDo.json')

# print("starting testing")
# start_time = time.time()

# results = utils.test_accuracy(net, testloader)
# tot_time = time.time()-start_time
# print("Acc: ", results)
# print("Time: ", tot_time)

if __name__ == '__main__':
    # inspect()
    # test_lib()
    # test_lib('./profiles/McDo.json')
    # test_vs()
    test_dset_zerop()
    test_dset_trainedp()

"""
probs = np.logspace(-0.1, -2.5, 20)
print(probs)

clean_accuracy, pert_accuracy, acti_accuracy, both_accuracy = utils.generate_graphs(
    net, testloader, probs)

plt.figure(1)
plt.plot(probs, clean_accuracy, probs, pert_accuracy,
         probs, acti_accuracy, probs, both_accuracy)
plt.title("Accuracy with regard to probability")
plt.xlabel("probability")
plt.ylabel("accuracy")
plt.legend(["clean", "only weights", "only activations", "weights and actis"])
plt.show()
"""
