"""A small barebone trainer for quick experiments."""
from FaultyMemory.utils.MetricManager import MetricManager
from typing import Callable, Dict, Tuple
from FaultyMemory.utils.DataHolder import DataHolder
from FaultyMemory.utils.accuracy import accuracy
from FaultyMemory import Handler
import torch
import logging

logger = logging.getLogger(__name__)

DEFAULT_OPT_PARAMS = {
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
}


class Trainer:
    def __init__(
        self,
        handler: Handler,
        dataholder: DataHolder,
        opt_criterion: Callable,
        device: torch.device,
        optim_params: Dict = DEFAULT_OPT_PARAMS,
        to_csv: bool = False,
    ) -> None:
        """Base trainer for one device, tested for simple (image, target) datasets.

        Args:
            handler (FyM.Handler): the handler to the quantized model
            dataholder (DataHolder): the dataset holder
            opt_criterion (Callable): a callable that compute the function to optimize
            device (torch.Device): the device to use
            optim_params (Dict, optional): parameters for the optimizer. Defaults to DEFAULT_OPT_PARAMS.
        """
        self.extra_information = {}
        self.handler = handler
        self.dataholder = dataholder
        self.metrics = MetricManager()
        self.opt_criterion = opt_criterion.to(device)
        self.device = device
        self.to_csv = to_csv
        self.init_optimizer(optim_params)
        self.train_loader, self.test_loader = self.dataholder.access_dataloader()

    @property
    def _information(self):
        """If some information changes, this ensures its reflected."""
        max_energy, current_energy = self.handler.energy_consumption()
        
        if hasattr(self.handler.net._hyperparameters):  
            hyperparameters = self.handler.net._hyperparameters
        else:
            logger.warn(
                "No hyperparameters were found on the network. "
                + "Please use log_hyperparameter class decorator on the network."
            )
            hyperparameters = {}

        if hasattr(self.optimizer._hyperparameters):  #TODO add optimizer
            hyperparameters |= self.optimizer._hyperparameters
        else:
            logger.warn(
                "No hyperparameters were found on the optimizer. "
                + "Please use log_hyperparameter class decorator on the network."
            )

        return {
            "dataset": self.dataholder.name,
            "architecture": self.handler.net.__class__.__name__,
            "max_energy_consumption": max_energy,
            "current_energy_consumption": current_energy,
            **hyperparameters,
        }

    def init_optimizer(self, optim_params: dict) -> None:
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.handler.net.parameters()),
            **optim_params
        )

    def loop(self, test_set: bool = True, grads_enabled: bool = True):
        assert (
            not test_set or test_set and not grads_enabled
        ), "Training on the test set is not cool :c"
        self.metrics.reset()
        self.extra_information["set"] = "test" if test_set else "train"

        if grads_enabled:
            self.handler.train()
        else:
            self.handler.eval()

        if test_set:
            dataloader = self.test_loader
        else:
            dataloader = self.train_loader

        with torch.set_grad_enabled(grads_enabled):
            self._loop(dataloader, grads_enabled)
        if self.to_csv:
            self.metrics.to_csv(self._information, self.extra_information)

    def _loop(self, dataloader: torch.utils.data.DataLoader, train_mode: bool = True):
        for i, sample in enumerate(dataloader):
            if (
                train_mode and "ticks" in self.extra_information
            ):  # TODO better handling of modes
                self.extra_information["ticks"] += 1

            self.handler.perturb_tensors()
            loss, metrics = self.forward(sample)
            metrics.update({"loss": loss.item()})
            self.metrics.log(metrics)

            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()

            self.handler.restore()
            if train_mode:
                self.optimizer.step()

    def forward(self, sample: Tuple) -> Tuple[torch.Tensor, dict]:
        """Return the differentiable loss and a dict with optionnal scalar metrics to be logged.
        Subclass at will for e.g. Mixup. Loss needn't be in the dict.
        Args:
            sample (Tuple): The sample to be processed by the neural network.

        Returns:
            Tuple[torch.Tensor, dict]: [description]

        """
        (images, targets) = sample
        images, targets = images.to(self.device), targets.to(self.device)
        output = self.handler.net(images)
        loss = self.opt_criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        return loss, {"acc1": acc1.item(), "acc5": acc5.item()}

    def train_loop(self) -> None:
        self.loop(False)

    def train_nograd_loop(self) -> None:
        self.loop(False, False)

    def test_loop(self) -> None:
        self.loop(True, False)

    def training_setup(self, epoch: int) -> None:
        """Setup tools and informations for the start of training.

        Args:
            epoch (int): Epoch at which the training (re)starts
        TODO set a scheduler by default
        """
        self.extra_information["epoch"] = epoch
        self.extra_information["ticks"] = 0  # The number of minibatches trained on

    def training_update(self) -> None:
        self.extra_information["epoch"] += 1

    def epoch_loop(self, nb: int, starting_epoch: int = 0):
        self.training_setup(starting_epoch)
        for _ in range(nb):
            self.training_update()
            self.train_loop()
            self.test_loop()

    def save(self):
        pass

    def load(self):
        pass
