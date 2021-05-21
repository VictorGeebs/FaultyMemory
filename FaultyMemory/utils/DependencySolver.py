"""If some classes depends on another, we need to know the dependency tree."""
import inspect
from typing import Callable
from FaultyMemory.utils.Checkpoint import Dependency


class DependencySolver:
    def __init__(self):
        self._items = {}

    def add_item(self, depending_class: Callable, dep_descr: Dependency):
        self._items.update({depending_class: dep_descr})

    def build_dep(self) -> list:
        """Return an ordered list of the classes to call in order to respect dependancies.

        Returns:
            list: [description]
        """
        assert self._constraint_satisfiable(), 'Some constraint could not be verified'
        satisfied, res = set(), []
        while(len(res) != len(self._items)):
            next_rank = self._next_rank(satisfied, res)
            satisfied.union(self._mro_satisfied(next_rank))
            res.append(next_rank)
        return res

    def _constraint_satisfiable(self) -> bool:
        mro, reqs = set(), set()
        for dep_class, deps in self._items():
            mro.union(set(inspect.getmro(type(dep_class))))
            reqs.union(set(deps.reqs))
        return all([reqs in mro])

    def _next_rank(self, satisfied: set, already: list = []) -> list:
        # TODO something like an orderedset to in order to not pass already and check it
        res = set()
        for dep_class, deps in self._items():
            if dep_class in already:
                continue
            if all([x in satisfied for x in deps.reqs]):
                res.add(dep_class)
        return list(res)

    def _mro_satisfied(self, satisfied: list) -> set:
        return set([inspect.getmro(type(x)) for x in satisfied])
