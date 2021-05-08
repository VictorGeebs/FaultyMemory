import torch
import copy
import re
import math
from numbers import Number
import numpy as np
import tqdm


def listify(obj: object) -> list:
    if isinstance(obj, list):
        return obj
    elif obj is None:
        return []
    else:
        return [obj]


def dictify(obj: object) -> dict:
    if isinstance(obj, dict):
        return obj
    elif obj is None:
        return {}
    elif isinstance(obj, list):
        return {str(type(o).__name__): o for o in obj}
    else:
        return {str(type(obj).__name__): obj}


def typestr_to_type(obj):
    res = re.match(r"\.([A-z]*)'", str(type(obj)))
    if res:
        return res.group(1)
    else:
        res = re.match(r"'([A-z]*)'", str(type(obj)))
        return res.group(1)


def ten_exists(model_dict: dict, name: str) -> None:
    try:
        assert (
            name in model_dict
        ), f"Specified name {name} not in model_dict and no reference given, cannot add this tensor"
    except AssertionError as id:
        print(id)
        return


def sanctify_ten(ten: torch.Tensor) -> torch.Tensor:
    r"""Take any tensor, make a deepcopy of it, and ensure the deepcopy is on CPU

    Note: `deepcopy` is needed in case the tensor is already on CPU, in which case `.cpu()` do not make a copy
    """
    return copy.deepcopy(ten.clone().detach()).cpu()


def sanitize_number(
    value: Number,
    mini: Number = float("-inf"),
    maxi: Number = math.inf,
    rnd: bool = False,
) -> Number:
    if rnd:
        value = round(value)
    return mini if value < mini else maxi if value > maxi else value


def kmeans_nparray(np_array: np.array, nb_clusters: int) -> np.array:
    from scipy.cluster.vq import kmeans, vq, whiten

    whitened = whiten(np_array)
    codebook, _ = kmeans(whitened, nb_clusters)
    encoded, _ = vq(np_array, codebook)
    return np.array([codebook[i] for i in encoded])


def test_accuracy(net, testloader) -> float:
    """
    A basic test function to test the accuracy of a network. \n
    This function might need modification depending on the type of label you
    wish to have.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            samples, labels = data
            samples, labels = samples.to(net.device), labels.to(net.device)
            outputs = net(samples)
            # net.restore()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print("running acc: ", correct/total)
            # break
    accuracy = correct / total
    return accuracy


def train_net(net, optimizer, criterion, trainloader, nb_epochs, prt=True):
    """
    A basic function used to train networks in a generic fashion
    """
    for epoch in range(nb_epochs):
        running_loss = 0.0
        print("epoch %i of %i" % (epoch + 1, nb_epochs))
        for i, data in enumerate(trainloader):

            inputs, labels = data

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if prt == True:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / (i + 1)))
    if prt == True:
        print("Finished Training")


@torch.jit.script
def twos_compl(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_not(tensor.abs()) + 1
