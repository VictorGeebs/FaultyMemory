"""If some classes depends on another, we need to know the dependency tree."""
import inspect
import copy
from typing import Callable
from FaultyMemory.utils.Checkpoint import Dependancy


class DependencySolver:
    def __init__(self):
        self._items = {}

    def add_item(self, depending_class: Callable, dep_descr: Dependancy):
        self._items.update({depending_class: dep_descr})

    def build_dep(self) -> list:
        """Return an ordered list of the classes to call in order to respect dependancies.

        Returns:
            list: [description]
        """
        assert self._constraint_satisfiable(), "Some constraint could not be verified"
        satisfied, res = [], []
        to_empty = copy.deepcopy(self._items)
        while len(to_empty) > 0:
            next_rank = self._next_rank(satisfied)
            satisfied.append(self._mro_satisfied(next_rank))
            [to_empty.pop(x) for x in next_rank]
            res.append(next_rank)
        return res

    def _constraint_satisfiable(self) -> bool:
        mro, reqs = [], []
        for dep_class, deps in self._items():
            mro.append(inspect.getmro(type(dep_class)))
            reqs.append(deps.reqs)
        return all([reqs in mro])

    def _next_rank(self, satisfied: list) -> list:
        res = []
        for dep_class, deps in self._items():
            if all([x in satisfied for x in deps.reqs]):
                res.append(dep_class)
        return res

    def _mro_satisfied(self, satisfied: list) -> list:
        pass
