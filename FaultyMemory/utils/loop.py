from typing import Dict, Tuple
from FaultyMemory.utils.DataHolder import DataHolder
import FaultyMemory as FyM
import torch


class Trainer:
    def __init__(
        self,
        handler: FyM.Handler,
        dataholder: DataHolder,
        criterion: torch.nn.Module,
        device: torch.nn.Device,
        optim_params: Dict,
    ) -> None:
        self.handler = handler
        self.dataholder = dataholder
        self.criterion = criterion
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

    def _loop(self, dataloader: torch.data.utils.Dataloader, train_mode: bool = True):
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
        loss = self.criterion(images, targets)
        return output, loss

    def epoch_loop(self, nb: int):
        for _ in range(nb):
            self.loop(False, True)
            self.loop(True, False)
