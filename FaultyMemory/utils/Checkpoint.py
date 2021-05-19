"""Manage and save these pesky overparameterized NN.

Kudos to the clever SpeechBrain gang https://github.com/speechbrain from where lots of good design ideas/practices were snatched.
"""
from FaultyMemory.handler import Handler
import collections
import functools
import inspect
import os
import pathlib
import torch
import yaml
import FaultyMemory
from typing import Any, Callable, List, Union
from datetime import datetime


ARGS_SKIP_LIST = ["progress", "self"]
OUTPUT_SIZE_ALIAS = ["num_classes"]
PRETRAINED_ALIAS = ["pretrained"]

CKPT_EXT = '.pthfaulty'
CKPT_PREFIX = 'CKPT'
META_FNAME = 'HPARAMS.yaml'


def log_hyperparameters(method_or_class):
    """A decorator to save the called method or class hyperperparameters in an attribute.
    Defaults (i.e. non changed) hyperparams are not logged.
    TODO monkey patching torch.nn.Module and optim.Optimizer in the init of the library to ensure its done ?
    """
    if inspect.isclass(method_or_class):

        @functools.wraps(method_or_class, updated=())
        class UpdatedCls(method_or_class):
            def __init__(self, *args, **kwargs) -> None:
                argnames = method_or_class.__init__.__code__.co_varnames[
                    1 : method_or_class.__init__.__code__.co_argcount
                ]
                argnames = dict(zip(argnames, args))
                [argnames.pop(k, None) for k in ARGS_SKIP_LIST]
                super().__init__(*args, **kwargs)
                self._hyperparameters = {**argnames, **kwargs}
        UpdatedCls._hparam_logger = True
        return UpdatedCls
    else:

        def updated_method(*args, **kwargs):
            argnames = method_or_class.__code__.co_varnames[
                : method_or_class.__code__.co_argcount
            ]
            argnames = dict(zip(argnames, args))
            [argnames.pop(k, None) for k in ARGS_SKIP_LIST]
            res = method_or_class(*args, **kwargs)
            res._hyperparameters = {**argnames, **kwargs}
            return res
        updated_method._hparam_logger = True
        return updated_method


def get_default_hook(obj: Any, default_hooks: dict):
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    return None


def pytorch_load(torch_object: Callable, path: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    save = torch.load(path, map_location=device)
    if not hasattr(torch_object, '_hparam_logger'):
        torch_object = log_hyperparameters(torch_object)
    model = torch_object(**save["hparams"]).to(device)
    model.load_state_dict(save["model"])
    return model


def pytorch_save(torch_object: Union[torch.nn.Module, torch.optim.Optimizer], path: str) -> dict:
    save = {'hparams': {}}
    if hasattr(torch_object, '_hyperparameters'):
        save['hparams'] = torch_object._hyperparameters
    save['model'] = torch_object.state_dict()
    torch.save(save, path)
    return save['hparams']


DEFAULTS_SAVE = {
    torch.nn.Module: pytorch_save,
    torch.optim.Optimizer: pytorch_save,
    FaultyMemory.Handler: Handler.dict_to_json,
}
DEFAULTS_LOAD = {
    torch.nn.Module: pytorch_load,
    torch.optim.Optimizer: pytorch_load,
    FaultyMemory.Handler: Handler.dict_from_json,
}


Checkpoint = collections.namedtuple(
    "Checkpoint", ["path", "meta", "paramfiles"]
)
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
            path = ckpt_dir_extended / f'{CKPT_PREFIX}_{name}{CKPT_EXT}'
            default_hook = get_default_hook(obj, DEFAULTS_SAVE)
            if default_hook is not None:
                hparams = default_hook(obj, path)
                extra_inf.update(hparams)
                continue
            raise TypeError(f"No saving method for {type(obj)}")
        self._save_meta(ckpt_dir_extended / f'{META_FNAME}', extra_inf)

    def _default_extended_path(self) -> pathlib.Path:
        # TODO check if already exists ?
        now = datetime.now().strftime("%d-%m-%y-%H-%M")
        return self.ckpt_dir / f'{CKPT_PREFIX}_{now}'

    def _save_meta(self, path: pathlib.Path, extra_inf: dict = {}) -> None:
        meta = {'time': datetime.now(), **extra_inf}
        with open(path, "w") as fo:
            fo.write("# yamllint disable\n")
            fo.write(yaml.dump(meta))

    # LOAD THINGIES
    def list_ckpt(self) -> List[Checkpoint]:
        return [self._construct_ckpt_objects(x) for x in self.ckpt_dir.iterdir() if Checkpointer.is_ckpt_dir(pathlib.Path(x))]

    @staticmethod
    def is_ckpt_dir(path: pathlib.Path) -> bool:
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / META_FNAME).exists()

    @staticmethod
    def _construct_ckpt_objects(ckpt_dir: str) -> Checkpoint:
        # This internal method takes a list of individual checkpoint
        # directory paths (as produced by _list_checkpoint_dirs)
        with open(ckpt_dir / META_FNAME) as fi:
            meta = yaml.load(fi, Loader=yaml.Loader)
        paramfiles = {}
        for ckptfile in ckpt_dir.iterdir():
            if ckptfile.suffix == CKPT_EXT:
                paramfiles[ckptfile.stem] = ckptfile
        return Checkpoint(ckpt_dir, meta, paramfiles)




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
