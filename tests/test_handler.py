import FaultyMemory as FyM
import pytest
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


@pytest.fixture
def simple_handler() -> FyM.Handler:
    net = torch.nn.Linear(10, 1)
    handler = FyM.Handler(net)
    handler(torch.ones((10)))
    return handler


def test_unparameterized_handler(simple_handler):
    e_max, e = simple_handler.energy_consumption()
    assert e_max == e == 176

