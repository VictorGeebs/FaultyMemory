"""Tests for representation.py."""

import FaultyMemory as FyM
import torch
import pytest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_tensor() -> torch.Tensor:
    """A simple 2 dimensional tensor put on device

    Returns:
        torch.Tensor: tensor of symmetric values around 0 from 1 to 3 in (2x3) format
    """
    return torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).to(device)


def encode_decode(tensor: torch.Tensor, representation: FyM.Representation):
    encoded = representation.encode(tensor)
    return encoded, representation.decode(encoded)


def test_freebie(simple_tensor) -> None:
    representation = FyM.FreebieQuantization()
    encoded, decoded = encode_decode(simple_tensor, representation)
    assert torch.equal(simple_tensor, encoded)
    assert torch.equal(simple_tensor, decoded)


def test_binary(simple_tensor) -> None:
    pass


def test_scaled_binary(simple_tensor) -> None:
    pass


def test_fixed_point(simple_tensor) -> None:
    pass


def test_ufixed_point(simple_tensor) -> None:
    pass


def test_slowfixed_point(simple_tensor) -> None:
    pass


def test_uslowfixed_point(simple_tensor) -> None:
    pass


def test_clustered(simple_tensor) -> None:
    pass


def test_distinct_reference(simple_tensor) -> None:
    r"""All encode and decode steps should produce a distinct tensor
    Device should be kept all along
    Input tensor and decoded tensor should have same dtype
    Also check if encoded is int8 (if digital)
    """
    for name, representation in FyM.REPR_DICT.items():
        print(name)
        instance = representation()
        encoded, decoded = encode_decode(simple_tensor, instance)
        assert decoded.dtype == simple_tensor.dtype
        assert encoded.device == simple_tensor.device == decoded.device
        assert simple_tensor.data_ptr() != encoded.data_ptr()
        assert simple_tensor.data_ptr() != decoded.data_ptr()
        assert encoded.shape == simple_tensor.shape == decoded.shape
        if instance.__COMPAT__ != "DIGITAL":
            continue
        assert encoded.dtype == torch.uint8
