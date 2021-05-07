"""A small barebone trainer for quick experiments."""
from typing import Callable, Dict, Tuple
from FaultyMemory.utils.DataHolder import DataHolder
from FaultyMemory import Handler
import torch

DEFAULT_OPT_PARAMS = {'lr': 0.1, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 5e-4}


class Trainer:
    def __init__(
        self,
        handler: Handler,
        dataholder: DataHolder,
        opt_criterion: Callable,
        device: torch.device,
        optim_params: Dict = DEFAULT_OPT_PARAMS,
    ) -> None:
        """Base trainer for one device, tested for simple (image, target) datasets.

        Args:
            handler (FyM.Handler): the handler to the quantized model
            dataholder (DataHolder): the dataset holder
            opt_criterion (Callable): a callable that compute the function to optimize
            device (torch.Device): the device to use
            optim_params (Dict, optional): parameters for the optimizer. Defaults to DEFAULT_OPT_PARAMS.
        """
        self.handler = handler
        self.dataholder = dataholder
        self.opt_criterion = opt_criterion.to(device)
        self.device = device
        self.init_optimizer(optim_params)
        self.train_loader, self.test_loader = self.dataholder.access_dataloader()

    def init_optimizer(self, optim_params: dict) -> None:
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.handler.net.parameters()),
            **optim_params
        )

    def loop(self, test_set: bool = True, grads_enabled: bool = True):
        assert (
            not test_set or test_set and not grads_enabled
        ), "Training on the test set is not cool :c"
        if grads_enabled:
            self.handler.train()
        else:
            self.handler.eval()

        if test_set:
            dataloader = self.test_loader
        else:
            dataloader = self.train_loader

        with torch.set_grad_enabled(grads_enabled):
            return self._loop(dataloader, grads_enabled)

    def _loop(self, dataloader: torch.utils.data.DataLoader, train_mode: bool = True):
        for i, sample in enumerate(dataloader):
            self.handler.perturb_tensors()
            output, loss = self.forward(sample)
            # TODO logging

            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()

            self.handler.restore()
            if train_mode:
                self.optimizer.step()

    def forward(self, sample: Tuple):
        (images, targets) = sample
        images, targets = images.to(self.device), targets.to(self.device)
        output = self.handler.net(images)
        loss = self.opt_criterion(output, targets)
        return output, loss

    def train_loop(self) -> None:
        self.loop(False)

    def test_loop(self) -> None:
        self.loop(True, False)

    def epoch_loop(self, nb: int):
        for _ in range(nb):
            self.train_loop()
            self.test_loop()
