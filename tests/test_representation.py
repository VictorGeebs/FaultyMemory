"""Tests for representation.py."""

import FaultyMemory as FyM
import torch
import pytest
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rounding_tensor() -> torch.Tensor:
    """A simple tensor put on device with values expected to be rounded

    Returns:
        torch.Tensor: tensor
    """
    return torch.tensor([3.20, 3.25, 3.5, 3.75]).to(device)


@pytest.fixture
def simple_tensor() -> torch.Tensor:
    """A simple 2 dimensional tensor put on device

    Returns:
        torch.Tensor: tensor of symmetric values around 0 from 1 to 3 in (2x3) format
    """
    return torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).to(device)


@pytest.fixture
def floating_tensor() -> torch.Tensor:
    """A simple 2 dimensional tensor put on device

    Returns:
        torch.Tensor: tensor of symmetric values around 0 from 0 to 1 (2x3) format
    """
    return torch.tensor([[-1.0, -0.5, -0.499999], [1.0, 0.5, 0.499999]]).to(device)


@pytest.fixture
def large_tensor() -> torch.Tensor:
    """A simple 2 dimensional tensor put on device

    Returns:
        torch.Tensor: tensor of symmetric values around 0 from 3 to 256 in (2x3) format
    """
    return torch.tensor([[-3.0, -128.0, -256.0], [3.0, 128.0, 256.0]]).to(device)


def encode_decode(tensor: torch.Tensor, representation: FyM.Representation):
    encoded = representation.encode(tensor)
    return encoded, representation.decode(encoded)


def test_freebie(simple_tensor) -> None:
    representation = FyM.FreebieQuantization()
    encoded, decoded = encode_decode(simple_tensor, representation)
    assert torch.equal(simple_tensor, encoded)
    assert torch.equal(simple_tensor, decoded)


def test_binary(simple_tensor) -> None:
    representation = FyM.BinaryRepresentation()
    encoded, decoded = encode_decode(simple_tensor, representation)
    target = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(decoded)
    ir = torch.tensor([[0, 0, 0], [1, 1, 1]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_scaled_binary(simple_tensor) -> None:
    representation = FyM.ScaledBinaryRepresentation()
    encoded, decoded = encode_decode(simple_tensor, representation)
    target = torch.tensor([[-2, -2, -2], [2, 2, 2]]).to(decoded)
    ir = torch.tensor([[0, 0, 0], [1, 1, 1]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_fixed_point_negativerange(caplog) -> None:
    with caplog.at_level(logging.INFO):
        representation = FyM.FixedPointRepresentation()
        representation.adjust_fixed_point(mini=-129, maxi=0)
        assert "Saturated range" in caplog.text


def test_fixed_point_positiverange(caplog) -> None:
    with caplog.at_level(logging.INFO):
        representation = FyM.FixedPointRepresentation()
        representation.adjust_fixed_point(mini=0, maxi=128)
        assert "Saturated range" in caplog.text


def test_fixed_point_range(caplog) -> None:
    with caplog.at_level(logging.INFO):
        representation = FyM.FixedPointRepresentation()
        representation.adjust_fixed_point(mini=-128, maxi=127)
        assert "Saturated range" in caplog.text


def test_fixed_point_floating_range(caplog) -> None:
    with caplog.at_level(logging.INFO):
        representation = FyM.FixedPointRepresentation()
        representation.adjust_fixed_point(mini=-64.5, maxi=63.5)
        assert "Saturated range" not in caplog.text


@pytest.mark.parametrize("type_repr", ["FixedPoint", "SlowFixedPoint"])
def test_fixed_point(floating_tensor, type_repr) -> None:
    representation = getattr(FyM, f"{type_repr}Representation")()
    encoded, decoded = encode_decode(floating_tensor, representation)
    target = torch.tensor([[-1, -0.5, -0.5], [1, 0.5, 0.5]]).to(decoded)
    ir = torch.tensor([[224, 240, 240], [32, 16, 16]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_ufixed_point_round(rounding_tensor) -> None:
    representation = FyM.UFixedPointRepresentation(width=3, nb_digits=0)  # i.e. Relu 6
    # [3.20, 3.25, 3.5, 3.75]
    encoded, decoded = encode_decode(rounding_tensor, representation)
    target = torch.tensor([3, 3, 4, 4]).to(decoded)
    ir = torch.tensor([3, 3, 4, 4]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)

    representation = FyM.UFixedPointRepresentation(
        width=4, nb_digits=1
    )  # i.e. Relu 6 + 1 bit
    encoded, decoded = encode_decode(rounding_tensor, representation)
    target = torch.tensor([3, 3.5, 3.5, 4]).to(decoded)
    assert torch.equal(decoded, target)


@pytest.mark.parametrize("type_repr", ["FixedPoint", "SlowFixedPoint"])
def test_fixed_point_saturated(large_tensor, type_repr) -> None:
    representation = getattr(FyM, f"{type_repr}Representation")()
    encoded, decoded = encode_decode(large_tensor, representation)
    target = torch.tensor([[-3, -4, -4], [3, 3.96875, 3.96875]]).to(decoded)
    ir = torch.tensor([[160, 128, 128], [96, 127, 127]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_ufixed_point(floating_tensor) -> None:
    representation = FyM.UFixedPointRepresentation()
    encoded, decoded = encode_decode(floating_tensor, representation)
    target = torch.tensor([[0, 0, 0], [1, 0.5, 0.5]]).to(decoded)
    ir = torch.tensor([[0, 0, 0], [32, 16, 16]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_ufixed_point_saturated(large_tensor) -> None:
    representation = FyM.UFixedPointRepresentation()
    encoded, decoded = encode_decode(large_tensor, representation)
    target = torch.tensor([[0, 0, 0], [3, 7.96875, 7.96875]]).to(decoded)
    ir = torch.tensor([[0, 0, 0], [96, 255, 255]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


def test_clustered(simple_tensor) -> None:
    import numpy as np

    np.random.seed(0)  # Ensure reproducibility
    [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]
    representation = FyM.ClusteredRepresentation(num_cluster=2)
    encoded, decoded = encode_decode(simple_tensor, representation)
    target = torch.tensor([[-2, -2, -2], [2, 2, 2]]).to(decoded)
    ir = torch.tensor([[1, 1, 1], [0, 0, 0]]).to(torch.uint8)
    assert torch.equal(encoded, ir)
    assert torch.equal(decoded, target)


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


def test_width() -> None:
    for w in range(1, 8):
        representation = FyM.UFixedPointRepresentation(width=w)
        assert representation.width == w
        representation = FyM.FixedPointRepresentation(width=w)
        assert representation.width == w
        representation = FyM.SlowFixedPointRepresentation(width=w)
        assert representation.width == w
    representation = FyM.BinaryRepresentation()
    assert representation.width == 1
    representation = FyM.ScaledBinaryRepresentation()
    assert representation.width == 1
    representation = FyM.ClusteredRepresentation()
    assert representation.width == 2
