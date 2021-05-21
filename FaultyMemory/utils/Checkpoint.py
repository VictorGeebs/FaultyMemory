"""Manage and save these pesky overparameterized NN.

Kudos to the clever SpeechBrain gang https://github.com/speechbrain from where lots of good design ideas/practices were snatched.
"""
from FaultyMemory.utils.DependencySolver import DependencySolver, Dependency
from FaultyMemory.handler import Handler
import collections
import inspect
import os
import pathlib
import torch
import yaml
import FaultyMemory
from FaultyMemory.utils.log_hparams import log_hyperparameters
from typing import Any, Callable, List, Union
from datetime import datetime


CKPT_EXT = ".pthfaulty"
CKPT_PREFIX = "CKPT"
META_FNAME = "HPARAMS.yaml"
TORCH_RECOVERABLE = Union[torch.nn.Module, torch.optim.Optimizer]


def get_default_hook(obj: Any, default_hooks: dict):
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    return None


def pytorch_load(
    torch_object: Union[Callable, TORCH_RECOVERABLE],
    path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> TORCH_RECOVERABLE:
    """Can instantiate or load on an existing instance.

    Args:
        torch_object (Union[Callable]): [description]
        path (str): [description]
        device (torch.device, optional): [description]. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        [type]: [description]
    """
    save = torch.load(path, map_location=device)
    if type(torch_object) not in TORCH_RECOVERABLE:
        # Need to instantiate
        if not hasattr(torch_object, "_hparam_logger"):
            torch_object = log_hyperparameters(torch_object)
        torch_object = torch_object(**save["hparams"]).to(device)
    torch_object.load_state_dict(save["model"])
    return torch_object


def pytorch_save(torch_object: TORCH_RECOVERABLE, path: str) -> dict:
    save = {"hparams": {}}
    if hasattr(torch_object, "_hyperparameters"):
        save["hparams"] = torch_object._hyperparameters
    save["model"] = torch_object.state_dict()
    torch.save(save, path)
    return save["hparams"]


DEFAULTS_SAVE = {
    torch.nn.Module: pytorch_save,
    torch.optim.Optimizer: pytorch_save,
    FaultyMemory.Handler: Handler.to_json,
}
DEFAULTS_LOAD = {
    torch.nn.Module: pytorch_load,
    torch.optim.Optimizer: Dependency(pytorch_load, [torch.nn.Module]),
    FaultyMemory.Handler: Dependency(Handler.from_json, [torch.nn.Module]),
}


Checkpoint = collections.namedtuple("Checkpoint", ["path", "meta", "paramfiles"])
# Creating a hash allows making checkpoint sets
Checkpoint.__hash__ = lambda self: hash(self.path)


class Checkpointer:
    def __init__(self, ckpt_dir: str, saveable_dict: dict) -> None:
        self.ckpt_dir = pathlib.Path(ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.saveable = {}
        for key, value in saveable_dict.items():
            self.saveable[key] = value

    # SAVE THINGIES
    def save_checkpt(self, name: str = None, extra_inf: dict = {}) -> None:
        """Save the Checkpoint ! c:.

        Note that for convenience, if hparams are set on objects to save,
        they'll be added in the HPARAMS.yaml

        Args:
            name (str, optional): [description]. Defaults to None.
            extra_inf (dict, optional): [description]. Defaults to {}.

        Raises:
            TypeError: [description]
        """
        if name is None:
            ckpt_dir_extended = self._default_extended_path()
        else:
            ckpt_dir_extended = self.ckpt_dir / name
        os.makedirs(ckpt_dir_extended)
        for name, obj in self.saveable.items():
            path = ckpt_dir_extended / f"{CKPT_PREFIX}_{name}{CKPT_EXT}"
            default_hook = get_default_hook(obj, DEFAULTS_SAVE)
            if default_hook is not None:
                hparams = default_hook(obj, path)
                extra_inf.update({name: hparams})
                continue
            raise TypeError(f"No saving method for {type(obj)}")
        self._save_meta(ckpt_dir_extended / f"{META_FNAME}", extra_inf)

    def _default_extended_path(self) -> pathlib.Path:
        # TODO check if already exists ?
        now = datetime.now().strftime("%d-%m-%y-%H-%M")
        return self.ckpt_dir / f"{CKPT_PREFIX}_{now}"

    def _save_meta(self, path: pathlib.Path, extra_inf: dict = {}) -> None:
        meta = {"time": datetime.now(), **extra_inf}
        with open(path, "w") as fo:
            fo.write("# yamllint disable\n")
            fo.write(yaml.dump(meta))

    # LOAD THINGIES
    def list_ckpt(self) -> List[Checkpoint]:
        return [
            self._construct_ckpt_objects(x)
            for x in self.ckpt_dir.iterdir()
            if Checkpointer.is_ckpt_dir(pathlib.Path(x))
        ]

    @staticmethod
    def is_ckpt_dir(path: pathlib.Path) -> bool:
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / META_FNAME).exists()

    @staticmethod
    def _construct_ckpt_objects(ckpt_dir: str) -> Checkpoint:
        # This internal method takes individual checkpoint
        # directory paths (as produced by _list_checkpoint_dirs)
        with open(ckpt_dir / META_FNAME) as fi:
            meta = yaml.load(fi, Loader=yaml.Loader)
        paramfiles = {}
        for ckptfile in ckpt_dir.iterdir():
            if ckptfile.suffix == CKPT_EXT:
                paramfiles[ckptfile.stem] = ckptfile
        return Checkpoint(ckpt_dir, meta, paramfiles)

    def load_ckpt(self, ckpt: Checkpoint):
        deps = DependencySolver()
        for name, item in self.saveable.items():
            loadpath = ckpt.paramfiles[name]
            default_hook = get_default_hook(item, DEFAULTS_LOAD)
            if default_hook is not None and type(default_hook) is not Dependency:
                default_hook(item, loadpath)
                deps.add_solved(item)
                continue
            elif default_hook is not None:  # assume its dependency
                deps.register_item(item, default_hook, loadpath)
            # If we got here, no custom hook or registered default hook exists
            raise RuntimeError(
                f"Don't know how to load {type(item)}. Register default hook \
                    or add custom hook for this object."
            )
        deps.process()


# import torch.nn as nn
# import torchvision
# @log_hyperparameters
# class MyModule(nn.Module):
#    def __init__(self, dimin, bias=True, **kwargs):
#        super().__init__()
#        self.conv = nn.Conv2d(1,1,1,bias=False)

# mymod = MyModule(1,bias=False)
# print(MyModule.__name__)
# log_hyperparameters(torchvision.models.resnet18)(True, zero_init_residual=True)
