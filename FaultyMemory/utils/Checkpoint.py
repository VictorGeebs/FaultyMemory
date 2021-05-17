import functools
import inspect


SKIP_LIST = ["progress", "self"]


def log_hyperparameters(method_or_class):
    """A decorator to save the called method or class hyperperparameters.
    Defaults (i.e. non changed) hyperparams are not logged
    """
    if inspect.isclass(method_or_class):

        @functools.wraps(method_or_class, updated=())
        class UpdatedCls(method_or_class):
            def __init__(self, *args, **kwargs) -> None:
                argnames = method_or_class.__init__.__code__.co_varnames[
                    1 : method_or_class.__init__.__code__.co_argcount
                ]
                argnames = dict(zip(argnames, args))
                [argnames.pop(k, None) for k in SKIP_LIST]
                super().__init__(*args, **kwargs)
                self._hyperparameters = {**argnames, **kwargs}

        return UpdatedCls
    else:

        def updated_method(*args, **kwargs):
            argnames = method_or_class.__code__.co_varnames[
                : method_or_class.__code__.co_argcount
            ]
            argnames = dict(zip(argnames, args))
            [argnames.pop(k, None) for k in SKIP_LIST]
            res = method_or_class(*args, **kwargs)
            res._hyperparameters = {**argnames, **kwargs}
            return res

        return updated_method


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
