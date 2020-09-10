import torch.nn as nn
import numpy as np


class Representation:
    """
    Base class for custom representations
    This base class is used to represent ints and uints of arbitrary width, though capped at width 8 at the moment for memory reasons
    """

    def __init__(self, width=8, unsigned=True):
        self.width = width
        self.unsigned = unsigned

    def __str__(self):
        print("Representation")
        return ""

    def __repr__(self):
        return self.__str__()

    def convert_to_repr(self, value):
        """
        Converts a value to the representation and returns it as its numpy version
        """
        if self.unsigned:
            value = value % (pow(2, self.width))
            return np.uint8(value)
        else:
            value = value % (pow(2, self.width))
            if value >= (pow(2, self.width) / 2):
                value = value - pow(2, self.width)
        return np.int8(value)

    def apply_mask(self, value, mask):
        """
        Returns the XOR of the mask and the value, bitwise
        """
        return value ^ mask

    def apply_tensor_mask(self, tensor, mask):
        """
        A parallelised version of apply_mask, which returns the bitwise XOR of an entire tensor with a tensor mask
        """
        param = np.bitwise_xor(tensor, mask)
        return param

    def to_json(self):
        """
        Creates and returns a dictionnary with the necessary information to re-construct this instance
        """
        dict = {}
        dict["name"] = self.__class__.__name__
        dict["width"] = self.width
        dict["unsigned"] = self.unsigned
        return dict


class BinaryRepresentation(Representation):
    """
    Binary Representation can take two forms:
    Unsigned: 0 or 1
    Signed: -1 or 1
    """

    def __init__(self, unsigned=False, width=1):  # TODO width always one?
        super(BinaryRepresentation, self).__init__()
        self.width = 1
        self.unsigned = unsigned

    def convert_to_repr(self, value):
        if not self.unsigned:
            if value <= 0:
                value = -1
            else:
                value = 1
            # suggestion: value = - (value <= 0) + (value > 0)
            # les branchements sont généralement couteux sur la performance, on peut donc y préferer les formulations comme celle-ci
            # c'est une micro-optimisation, qu'il faudrait idéalement benchmarker sur le code en c
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
            return value * -1

    def apply_tensor_mask(self, tensor, mask):
        mask = mask.astype("int")
        mask = mask * -2 + 1
        param = tensor * mask
        return param


"""
This dictionnary is used to construct representations from a JSON input
"""
RepresentationDict = {
    "Representation": Representation,
    "BinaryRepresentation": BinaryRepresentation,
}

"""
Constructs a representation according to the dictionnary provided.
The dictionnary should have a field for 'name' equals to the name of the class, the width and wether or not it is unsigned.
"""


def construct_repr(repr_dict):
    if repr_dict is None:
        return None
    instance = RepresentationDict[repr_dict["name"]](
        width=repr_dict["width"], unsigned=repr_dict["unsigned"]
    )
    return instance