import FaultyMemory.representation as Representation
import torch


def encode_decode(tensor: torch.Tensor, repr: Representation):
    encoded = repr.encode(tensor)
    return encoded, repr.decode(encoded)


def test_freebie():
    repr = Representation.FreebieQuantization()
    tensor = torch.tensor([1.0])
    encoded, decoded = encode_decode(tensor, repr)
    assert torch.equal(tensor, encoded)
    assert torch.equal(tensor, decoded)


def test_distinct_reference():
    r"""All encode and decode steps should produce a distinct tensor
    Also check if encoded is int8 and device is kept
    """
    for repr in Representation.REPR_DICT.values():
        instance = repr()
        tensor = torch.tensor([1.0])
        encoded, decoded = encode_decode(tensor, instance)
        assert encoded.dtype == torch.uint8
        assert decoded.dtype == tensor.dtype
        assert encoded.device == tensor.device == decoded.device
        assert tensor.storage.data_ptr() != encoded.storage.data_ptr()
        assert tensor.storage.data_ptr() != decoded.storage.data_ptr()
