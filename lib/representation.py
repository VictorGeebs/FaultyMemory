import torch
import torch.nn as nn
import random
import numpy as np

class Representation():
    def __init__(self, width=8, unsigned=True):
        self.width = width
        self.unsigned = unsigned
    
    def convert_to_repr(self, value):
        if self.unsigned == True:
            value = value % (pow(2, self.width))
            return np.uint8(value)
        else:
            value = (value % (pow(2, self.width)))
            if (value >= (pow(2, self.width)/2)):
                value = value - pow(2, self.width)
        return np.int8(value)

    def apply_mask(self, value, mask):
        return value^mask

    def apply_tensor_mask(self, tensor, mask):
        param = np.bitwise_xor(tensor, mask)
        return param

class BinaryRepresentation(Representation):
    def __init__(self, unsigned=False, width=1):
        super(BinaryRepresentation, self).__init__()
        self.width = 1
        self.unsigned=unsigned

    def convert_to_repr(self, value):
        if self.unsigned == False:
            if value <= 0:
                value = -1
            else:
                value = 1
        else:
            if value >= 0.5:
                value = 1
            else:
                value = 0
        return value

    def apply_mask(self, value, mask):
        if mask == 0:
            return value
        else:
            return value*-1

    def apply_tensor_mask(self, tensor, mask):
        mask = mask.astype('int')
        mask = mask*-2 + 1
        param = tensor * mask
        return param

RepresentationDict = {
    "Representation": Representation,
    "BinaryRepresentation": BinaryRepresentation
}

def construct_repr(repr_dict):
    if repr_dict is None:
        return None
    instance = RepresentationDict[repr_dict['name']](width=repr_dict['width'], unsigned=repr_dict['unsigned'])
    return instance