from FaultyMemory.utils.misc import dictify, ten_exists, sanctify_ten
from FaultyMemory.perturbator import Perturbator, construct_pert
from typing import Callable, Optional, Tuple, Union
from FaultyMemory.representation import (
    FixedPointRepresentation,
    Representation,
    construct_repr,
)

from tabulate import tabulate
import copy
import numpy as np
import torch
import torch.nn as nn
import inspect
import logging
from abc import ABC, abstractclassmethod

TYPE_DICT = {}


logger = logging.getLogger(__name__)


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
        representation: Representation,
        pert: Optional[Union[dict, Perturbator]] = None,
    ) -> None:
        # FLAGS (PRIVATE)
        self._perturb = True
        self._restored = True

        # PROPERTY (PUBLIC)
        self.name = name
        self.repr = copy.deepcopy(representation)
        self.pert = dictify(pert)
        self.model = model
        self.callbacks = {}
        self.tensor_stats = {}

        # INIT REFS
        ten_exists(self.where_ten(), name)
        self.compute_bitcount()
        self.saved_ten = None
        self.save()

    def __str__(self) -> str:  # pragma: no cover
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

    def __del__(self) -> None:
        if not self._restored:
            self.restore()

    @classmethod
    def from_dict(cls, _dict: dict, model: nn.Module):
        return cls(
            model,
            _dict["name"],
            construct_repr(_dict["repr"]),
            {pert["name"]: construct_pert(pert) for pert in _dict["pert"]},
        )

    def access_ten(self) -> torch.Tensor:
        return self.where_ten()[self.name]

    @abstractclassmethod
    def where_ten(self) -> dict:
        pass

    def on_perturbs(self) -> None:
        self._perturb = True

    def off_perturbs(self) -> None:
        self._perturb = False

    def default_exec_callback_stack(self) -> None:
        r"""Run the callback stack immediately.

        Counterpart of quantize_perturb that do not ensure quantization/perturbation
        Mainly for testing purposes/RepresentedWeights
        """
        self.exec_callbacks_stack(self.access_ten())

    def exec_callbacks_stack(self, tensor) -> None:
        # TODO an assert that the tensor is not changed apart when priority == 0
        # TODO a parameter to control if all the callbacks needs to be executed ? e.g. only pre to_repr
        for name, callback in sorted(
            self.callbacks.items(), key=lambda x: x[1]["priority"]
        ):
            callback["func"](self, tensor)
            if callback["autoremove"]:
                del self.callbacks[name]

    def register_callback(
        self, callback: Callable, name: str, priority: float, autoremove: bool = False
    ) -> None:
        assert (
            len(inspect.signature(callback).parameters) == 2
        ), "A callback has a signature (self, tensor)"
        self.callbacks[name] = {
            "priority": priority,
            "func": callback,
            "autoremove": autoremove,
        }
        if not self._restored:
            print(
                "A callback was attached while the tensor is already quantized. Restore and quantize to perform the new callback."
            )
        # TODO an idea to dynamically re-execute the cb stack each time a new cb is attached. A pain with the saves though. worth it ?
        # if self._executed:
        #    self.restore()
        #    self.default_exec_callback_stack()

    def access_ten_before(
        self,
        callback: Callable,
        name: str,
        autoremove: bool = False,
    ) -> None:
        self.register_callback(callback, name, float("-inf"), autoremove)

    def access_ten_after(
        self, callback: Callable, name: str, autoremove: bool = False
    ) -> None:
        self.register_callback(callback, name, float("inf"), autoremove)

    def detach_callback(self, name: str, tentative: False):
        if name in self.callbacks:
            self.callbacks.pop(name)
        elif not tentative:
            print(f"The callback {name} is not registered")

    def to_repr(self, x) -> None:
        # TODO pre-compute the fault mask with bitwise ops
        # so as not to create dependencies in the comp.
        # graph on encoded and achieve potential speed-ups
        encoded = self.repr.encode(x)
        if self._perturb:
            encoded = self.apply_perturb_to_encoded(encoded)
        return self.repr.decode(encoded)

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
            assert self._restored, "Trying to quantize without prior restore"
            self._restored = False
            output.data.copy_(self.to_repr(output).data)

        self.register_callback(func, name="quantize_perturb", priority=0)

    def save(self) -> None:
        def func(self, ten) -> None:
            if self.saved_ten is None:
                self.saved_ten = sanctify_ten(ten)
                self.ref_ten = ten

        self.access_ten_before(func, name="save_tensor")

    def restore(self, purge: bool = True) -> None:
        """Restore the representend tensor to an original state.

        Args:
            purge (bool, optional): Delete the saved copy of the tensor if True. Defaults to True.
        """
        self._restored = True
        if self.saved_ten is not None and purge:
            del self.saved_ten
            self.saved_ten = None

    def compute_bitcount(self) -> None:
        def func(self, output) -> None:
            self.tensor_stats["bitcount"] = output.numel() * self.repr.width

        self.access_ten_after(func, name="compute_bitcount", autoremove=True)

    def energy_consumption(self, a=12.8) -> Tuple[int, float]:
        if "bitcount" not in self.tensor_stats.keys():
            logger.warning(
                f"Bitcount of {self.name} has not been set in `compute_bitcount`\
                        Bitcount is set to default = 0."
            )
            bitcount = 0
        else:
            bitcount = self.tensor_stats["bitcount"]

        def energy_formula(p):
            return -np.log(p) / a if (p > 0) & (p < 0.5) else 1 if p <= 0 else 0

        if "BernoulliXORPerturbation" in self.pert:
            p = self.pert["BernoulliXORPerturbation"].distribution.probs.cpu().numpy()
            if np.ndim(p) == 0:
                current_consumption = energy_formula(p)
            else:
                current_consumption = np.average([energy_formula(pi) for pi in p])
        else:
            logger.warning(
                f"There are no consumption model other than for BernoulliXORPerturbation yet.\
                Consumption of {self.name} is set to default = 1."
            )
            current_consumption = 1

        return (
            bitcount,
            bitcount * current_consumption,
        )

    def quantize_mse(self) -> None:
        r"""Computes the Mean Squared Error between an original tensor and its quantized version
        TODO quantify the impact of data movs. Maybe let saved_ten stay on device at first ?
        """

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
            "MAX" not in self.tensor_stats.keys()
            and "MIN" not in self.tensor_stats.keys()
            and "ValueRange" not in self.callbacks
            and "DynamicValueRange" not in self.callbacks
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

    def __init__(
        self,
        model: nn.Module,
        name: str,
        representation: Representation,
        pert: Optional[Union[dict, Perturbator]] = None,
    ) -> None:
        super().__init__(model, name, representation, pert=pert)
        self.default_exec_callback_stack()  # For bitcount
        super().restore()  # Weird timing behavior: see below
        # FIXME
        # When the represented parameter is overwritten
        # Some unwanted timing between the previous __del__
        # and the current __init__ makes a copy of the
        # potentially previous to_repr in self.saved_ten
        # super().restore() here ensure there is no such copy

    def where_ten(self) -> dict:
        return dict(self.model.named_parameters())

    def quantize_perturb(self) -> None:
        super().quantize_perturb()
        self.default_exec_callback_stack()

    def restore(self, purge: bool = True) -> None:
        if self.saved_ten is not None:
            self.ref_ten.data.copy_(self.saved_ten.data.to(self.ref_ten))
        super().restore(purge=purge)


@add_type
class RepresentedActivation(RepresentedTensor):
    r"""Seamlessly cast an activation tensor to faulty hardware"""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        representation: Representation,
        pert: Optional[Union[dict, Perturbator]] = None,
    ) -> None:
        super().__init__(model, name, representation, pert=pert)
        self.hook = self.access_ten().register_forward_hook(
            self.exec_callbacks_stack_act
        )

    def exec_callbacks_stack_act(self, module, input, output) -> None:
        self.exec_callbacks_stack(output)

    def where_ten(self) -> dict:
        return dict(self.model.named_modules())

    def default_exec_callback_stack(self) -> None:
        # TODO perform the callback stack on self.saved_ten ~ useful ?
        pass

    def compute_bitcount(self) -> None:
        def func(self, output) -> None:
            self.tensor_stats["bitcount"] = (
                output.numel() / output.shape[0]
            ) * self.repr.width

        self.access_ten_after(func, name="compute_bitcount", autoremove=True)

    def restore(self, purge: bool = True) -> None:
        logging.info("Cannot restore an activation, will only delete the saved tensor")
        super().restore(purge=purge)

    def __del__(self):
        self.hook.remove()


class RepresentedModule_(RepresentedTensor):
    r"""Replace a module with a represented one
    TODO the goal is to not be seamless, i.e. the network definition changes
    """


def construct_type(model: nn.Module, type_dict: dict):
    return TYPE_DICT[type_dict.pop("type")].from_dict(type_dict, model)
