import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import FaultyMemory.handler as H
import FaultyMemory.representation as R
import FaultyMemory.perturbator as P
import FaultyMemory.utils as utils
from models import resnet, densenet, vgg
import FaultyMemory.represented_tensor as represented_tensor
import csv
import datetime
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# alexnet = models.alexnet(pretrained=True)
# resnet18 = models.resnet18(pretrained=True).to(device)
# resnet18.device = device
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)

resnet18 = resnet.resnet18(pretrained=True).to(device)
resnet18.device = device
resnet18.name = "resnet18"

densenet = densenet.densenet121(pretrained=True).to(device)
densenet.device = device
densenet.name = "densenet"

vgg16 = vgg.vgg16_bn(pretrained=True).to(device)
vgg16.device = device
vgg16.name = "vgg16"

net_list = [resnet18, densenet, vgg16]

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([transforms.ToTensor(), normalize])


testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=2)

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

repr_list = [R.FixedPointRepresentation(width=8, nb_digits=8)]
pert_list = [P.BernoulliXORPerturbation(probs=0), P.BernoulliXORPerturbation(probs=0.01), P.BernoulliXORPerturbation(probs=0.025), P.BernoulliXORPerturbation(probs=0.05), P.BernoulliXORPerturbation(probs=0.1)]

csv_data = [["Net Name", "Pert", "Repr", "Init acc", "Handler", "Init Trained", "Handler trained"]]


for net in net_list:
    for repr in repr_list:
        for pert in pert_list:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            net_copy = copy.deepcopy(net)
            handler = H.Handler(net_copy)
            handler.device = device
            pert.width = repr.width
            handler.add_net_parameters(representation=repr, perturb=pert)

            mse_list = handler.compute_MSE()
            min_mse_index = mse_list.index(min(mse_list))

            net_copy.eval()

            init_acc = utils.test_accuracy(net_copy, testloader)
            handler_acc = utils.test_accuracy(handler, testloader)
    
            net_copy.train()
            utils.train_net(handler, optimizer, criterion, trainloader, 5, prt=False)

            init_trained_acc = utils.test_accuracy(net_copy, testloader)
            handler_trained_acc = utils.test_accuracy(handler, testloader)

            csv_data.append([net.name, float(pert.distribution.probs), repr.width, init_acc, handler_acc, init_trained_acc, handler_trained_acc])

with open('test_nets_' + datetime.now().strftime("%d_%m_%Y__%H_%M_%S") + ".csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)