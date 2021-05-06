"""Off the shelf dataloaders for standard datasets/augments."""
from typing import Tuple
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


IMAGENET_NORMALIZE = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
)
DATASET_CLASS_COUNT = {"CIFAR10": 10, "CIFAR100": 100}


class DataHolder:
    def __init__(
        self,
        name: str = "CIFAR10",
        path: str = "./datasets",
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> None:
        """Base class that should works for all pytorch standard datasets.
        Subclass at will.
        TODO: find a nicer way to access the nb of classes
        """
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        _ = self.access_dataset()  # These ensures the dataset is available
        _ = self.class_count()

    def access_dataset(self) -> Tuple[Dataset, Dataset]:
        base = getattr(torchvision.datasets, self.name)
        kwargs = {
            "root": self.path,
            "download": True,
        }  # TODO: test if download is necessary
        return (
            base(train=True, transform=self.transform_train(), **kwargs),
            base(train=False, transform=self.transform_test(), **kwargs),
        )

    def transform_train(self):
        return transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ]
        )

    def transform_test(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ]
        )

    def class_count(self) -> int:
        if self.name in DATASET_CLASS_COUNT:
            return DATASET_CLASS_COUNT[self.name]
        else:
            raise ValueError("The dataset has no known class count")

    def access_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        trainset, testset = self.access_dataset()
        return (
            self._dataloader_factory(trainset),
            self._dataloader_factory(testset, False),
        )

    def _dataloader_factory(self, dataset, shuffle=True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
