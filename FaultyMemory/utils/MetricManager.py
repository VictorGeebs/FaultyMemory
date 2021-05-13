import os
from typing import Union

from numbers import Number
from FaultyMemory.utils.Metric import Metric
from pathlib import Path
import csv
import datetime
import copy


class MetricManager:
    def __init__(self) -> None:
        """A class that manages Metrics objects.
        TODO allow for different sampling rates between metrics (e.g. one log per epoch/one log per minibatch)
        TODO manage a logger (export to .csv)
        """
        self._metrics = {}

    def get_metric(self, name: str) -> Metric:
        if name in self._metrics:
            return self._metrics[name]
        else:
            raise ValueError("This Metric was not found in the records")

    def log(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            self._log_dict(args[0])
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], Number):
            self._log_scalar(*args)

    def _log_dict(self, value_dict: dict) -> None:
        for key, value in value_dict.items():
            self.log(key, value)

    def _log_scalar(self, name: str, value: Union[int, float]) -> None:
        if name in self._metrics:
            self._metrics[name].update(value)
        else:
            metric = Metric()
            metric.update(value)
            self._metrics[name] = metric

    def _apply(self, func_name: str):
        return {k: getattr(v, func_name)() for (k, v) in self._metrics.items()}

    def reset(self):
        """Reset to empty the metrics."""
        self._metrics = {}

    def average(self) -> dict:
        return self._apply("average")

    def to_csv(self, information: dict = {}, extra_information: dict = {}):
        assert 'dataset' in information, 'A name for the datasets used needs to be provided'
        information, extra_information = copy.deepcopy(information), copy.deepcopy(extra_information)
        if "set" in extra_information:
            filename = os.path.join(information.pop("dataset"), f'{extra_information.pop("set")}.csv')
        else:
            filename = f'{information.pop("dataset")}.csv'

        datapoints = self.average()
        if not Path(filename).exists():
            write_heads = True
        with open(filename, "a+") as f:
            writer = csv.writer(f)
            now = datetime.now()
            if write_heads:
                writer.writerow(
                    ["date", "architecture", "max_energy", "current_energy"]
                    + list(extra_information.keys())
                    + list(datapoints.keys())
                )
            writer.writerow(
                [
                    now,
                    information["architecture"],
                    information["max_energy_consumption"],
                    information["current_energy_consumption"],
                ]
                + list(extra_information.values())
                + list(datapoints.values())
            )
