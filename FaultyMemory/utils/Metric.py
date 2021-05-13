from typing import Union
import numpy as np


class Metric:
    def __init__(self, max_history: int = 1e3) -> None:
        self._hist = []
        self._max_history = max_history

    def reset(self):
        self._hist = []

    def non_empty(self):
        assert len(self._hist) > 0, "This Metric is empty"

    def update(self, value: Union[int, float]):
        assert isinstance(value, float) or isinstance(
            value, int
        ), "This Metric applies to scalar only"
        self.hist.append(value)
        if len(self.hist) > self._max_history:
            self.hist = self.hist[-self._max_history :]

    def last(self):
        self.non_empty()
        return self._hist[-1]

    def average(self):
        self.non_empty()
        return np.average(self._hist)
