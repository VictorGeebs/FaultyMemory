from FaultyMemory.utils import dictify, ten_exists, sanctify_ten
from FaultyMemory.perturbator import Perturbator, construct_pert
from typing import Callable, Optional, Tuple, Union
from FaultyMemory.representation import (
    FixedPointRepresentation,
    Representation,
    construct_repr,
)
from tabulate import tabulate
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractclassmethod

import copy

TYPE_DICT = {}


def add_type(func):
    TYPE_DICT[func.__name__] = func
    return func


class RepresentedTensor(ABC):
    r"""
    Works through a callback system, functions in the dict callbacks are stored as dict('priority', 'func')
    Priority ordering the execution from -inf to 0 (quantized) to inf
    """

    def __init__(
        self,
        model: nn.Module,
        name: str,
        repr: Representation,
        pert: Optional[Union[dict, Perturbator]] = {},
    ) -> None:
        self.name = name
        self._perturb = True
        self.repr = copy.deepcopy(repr)
        self.pert = dictify(pert)
        self.model = model
        ten_exists(self.where_ten(), name)
        self.compute_bitcount()
        self.saved_ten = None
        self.tensor_stats = {}
        self.callbacks = {}

    def __str__(self) -> str:
        string = f"{self.repr} tensor \n"
        string += tabulate(
            self.pert, ["Perturbation type", "Params."], tablefmt="github"
        )
        string += tabulate(self.tensor_stats, ["Stat.", "Value"], tablefmt="github")
        string += tabulate(
            sorted([(v["priority"], k) for k, v in self.callbacks.items()]),
            ["Priority", "Callback"],
            tablefmt="github",
        )
        return string

    @classmethod
    def from_dict(cls, dict: dict, model: nn.Module):
        return cls(
            model,
            dict["name"],
            construct_repr(dict["repr"]),
            {pert["name"]: construct_pert(pert) for pert in dict["pert"]},
        )

    def access_ten(self):
        return self.where_ten()[self.name]

    @abstractclassmethod
    def where_ten(self) -> dict:
        pass

    def on_perturbs(self) -> None:
        self._perturb = True

    def off_perturbs(self) -> None:
        self._perturb = False

    def default_exec_callback_stack(self) -> None:
        r"""Run the callback stack immediately
        Counterpart of quantize_perturb that do not ensure quantization/perturbation
        Mainly for testing purposes
        """
        self.exec_callbacks_stack(self.access_ten())

    def exec_callbacks_stack(self, tensor) -> None:
        # TODO an assert that the tensor is not changed apart when priority == 0
        # TODO a parameter to control if all the callbacks needs to be executed ? e.g. only pre to_repr
        for name, callback in sorted(
            self.callbacks.items(), key=lambda x: self.callbacks[x]["priority"]
        ):
            callback["func"](tensor)
            if callback["autoremove"]:
                del self.callbacks[name]

    def register_callback(
        self, callback: Callable, name: str, priority: float, autoremove: bool = False
    ) -> None:
        self.callbacks[name] = {
            "priority": priority,
            "func": callback,
            "autoremove": autoremove,
        }

    def access_ten_before(
        self,
        callback: Callable,
        autoremove: bool = False,
        name: str = "before_quantize",
    ) -> None:
        self.register_callback(callback, name, float("-inf"), autoremove)

    def access_ten_after(
        self, callback: Callable, autoremove: bool = False, name: str = "after_quantize"
    ) -> None:
        self.register_callback(callback, name, float("inf"), autoremove)

    def detach_callback(self, name: str):
        if name in self.callbacks:
            self.callbacks.pop(name)
        else:
            print(f"The callback {name} is not registered")

    def to_repr(self, x) -> None:
        encoded = self.repr.encode(x)
        assert (
            encoded.shape == x.shape
        ), "The encoded version is not of the same shape as the input tensor"
        if self._perturb:
            encoded = self.apply_perturb_to_encoded(encoded)
        return self.repr.decode(encoded).to(x.dtype)

    def apply_perturb_to_encoded(self, base) -> torch.Tensor:
        for _, pert in self.pert.items():
            if not pert:
                continue
            assert self.repr.compatibility(
                pert
            ), "The perturbation is not compatible with the representation"
            base = pert(base)
        return base

    def quantize_perturb(self) -> None:
        def func(self, output) -> None:
            output.data.copy_(self.to_repr(output).data)

        self.register_callback(func, name="quantize_perturb", priority=0)

    def save(self) -> None:
        def func(self, ten) -> None:
            if self.saved_ten is None:
                self.saved_ten = sanctify_ten(ten)
                self.ref_ten = ten
            else:
                # TODO test if 2 GPU trigger this print
                print("Another tensor is already saved")

        self.access_ten_before(func, name="save_tensor", autoremove=True)

    def restore(self, purge: bool = True) -> None:
        if self.saved_ten is not None:
            self.ref_ten.data.copy_(self.saved_ten.data.to(self.ref_ten))
            if purge:
                del self.saved_ten
                self.saved_ten = None

    def compute_bitcount(self) -> None:
        def func(self, output) -> None:
            self.tensor_stats["bitcount"] = output.numel() * self.repr.width

        self.access_ten_after(func, name="compute_bitcount", autoremove=True)

    def energy_consumption(self, a=12.8) -> Tuple[int, float]:
        assert (
            "bitcount" in self.tensor_stats.keys()
        ), "Bitcount has not been set in `compute_bitcount`"
        if "BernoulliXORPerturbation" in self.pert:
            p = self.pert["BernoulliXORPerturbation"].distribution.probs.cpu().numpy()
        else:
            print(
                "There are no consumption model other than for BernoulliXORPerturbation yet"
            )
            p = 0.0
        current_consumption = -np.log(p / a) if p > 0 else 1.0
        return (
            self.tensor_stats["bitcount"],
            self.tensor_stats["bitcount"] * current_consumption,
        )

    def quantize_MSE(self) -> None:
        r"""Computes the Mean Squared Error between an original tensor and its quantized version
        TODO quantify the impact of data movs. Maybe let saved_ten stay on device at first ?
        """
        if self.saved_ten is None and not "save_tensor" in self.callbacks:
            self.save()

        def func(self, output) -> None:
            ten = self.saved_ten.to(output)
            loss = nn.MSELoss().to(output)
            self.tensor_stats["MSE"] = loss(output, ten).item()
            self.saved_ten.to("cpu")

        self.register_callback(func, name="MSE", priority=1)

    def value_range(self) -> None:
        r"""Computes the range of values in the tensor to allow for smart representation assignation"""

        def func(self, output) -> None:
            self.tensor_stats["MAX"] = torch.max(output).item()
            self.tensor_stats["MIN"] = torch.min(output).item()

        self.access_ten_before(func, name="ValueRange")

    def dyn_value_range(self) -> None:
        r"""Computes the range of values in the tensor to allow for smart representation assignation"""

        def func(self, output) -> None:
            self.tensor_stats["MAX"] = max(
                torch.max(output).item(), self.tensor_stats.get("MAX", float("-inf"))
            )
            self.tensor_stats["MIN"] = min(
                torch.min(output).item(), self.tensor_stats.get("MIN", float("inf"))
            )

        self.access_ten_before(func, name="ValueRange")

    def adjust_fixed_point(self) -> None:
        r"""Adjust the representation parameters to the tensor statistics
        TODO generalize to repr others than fixed point
        """
        if (
            not "MAX" in self.tensor_stats.keys()
            and not "MIN" in self.tensor_stats.keys()
            and not "ValueRange" in self.callbacks
            and not "DynamicValueRange" in self.callbacks
        ):
            print(
                "Tensors statistics have yet to be computed, auto-attaching the `value_range` callback"
            )
            self.value_range()

        def func(self, output) -> None:
            if isinstance(self.repr, FixedPointRepresentation):
                self.repr.adjust_fixed_point(
                    self.tensor_stats["MIN"], self.tensor_stats["MAX"]
                )

        self.register_callback(func, name="TuneRepr", priority=-1, autoremove=True)

    def to_json(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "repr": self.repr.to_json(),
            "pert": [pert.to_json() for _, pert in self.pert.items()],
        }


@add_type
class RepresentedParameter(RepresentedTensor):
    r"""Seamlessly cast a parameter tensor to faulty hardware"""

    def where_ten(self) -> dict:
        return dict(self.model.named_parameters())

    def quantize_perturb(self) -> None:
        super().quantize_perturb()
        self.default_exec_callback_stack()


@add_type
class RepresentedActivation(RepresentedTensor):
    r"""Seamlessly cast an activation tensor to faulty hardware"""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        repr: Representation,
        pert: Optional[Union[dict, Perturbator]],
    ) -> None:
        super().__init__(model, name, repr, pert=pert)
        self.hook = self.access_ten().register_forward_hook(
            self.exec_callbacks_stack_act
        )

    def exec_callbacks_stack_act(self, module, input, output) -> None:
        self.exec_callbacks_stack(output)

    def where_ten(self) -> dict:
        return self.model.named_modules()

    def default_exec_callback_stack(self) -> None:
        pass

    def compute_bitcount(self) -> None:
        def func(self, output) -> None:
            self.tensor_stats["bitcount"] = (
                output.numel() / output.shape[0]
            ) * self.repr.width

        self.access_ten_after(func, name="compute_bitcount", autoremove=True)

    def restore(self, purge: bool = True) -> None:
        print("Cannot restore an activation, will only delete the saved tensor")
        if self.saved_ten is not None and purge:
            del self.saved_ten
            self.saved_ten = None

    def __del__(self):
        self.hook.remove()
        super().__del__()


class RepresentedModule_(RepresentedTensor):
    r"""Replace a module with a represented one
    TODO the goal is to not be seamless, i.e. the network definition changes
    """
    pass


def construct_type(model: nn.Module, type_dict: dict):
    return TYPE_DICT[type_dict.pop("type")].from_dict(type_dict, model)
