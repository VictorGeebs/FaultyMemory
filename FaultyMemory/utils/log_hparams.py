import functools
import inspect

ARGS_SKIP_LIST = ["progress", "self"]
OUTPUT_SIZE_ALIAS = ["num_classes"]
PRETRAINED_ALIAS = ["pretrained"]


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
                    1:method_or_class.__init__.__code__.co_argcount
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
