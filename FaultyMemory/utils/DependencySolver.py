"""If some classes depends on another, we need to know the dependency tree."""
import inspect
import copy
import collections
from typing import Any


Dependency = collections.namedtuple("Dependancy", ["build_class", "reqs"])


class DependencySolver:
    def __init__(self):
        """Take a dict of Dependency and generate an ordered list of these dependency."""
        self._items = {}
        self._solved = set()

    def register_item(self, depending_class: Any, dep_descr: Dependency, loadpath: str):
        dep_descr._loadpath = loadpath
        self._items.update({depending_class: dep_descr})

    def add_solved(self, solved: Any):
        solved = inspect.getmro(solved)
        self._solved.union(set(solved))

    def build_dep(self) -> list:
        """Return an ordered list of the Dependencies to call in order.

        Returns:
            list: [description]
        """
        assert self._constraint_satisfiable(), "Some constraint could not be verified"
        satisfied, res = copy.deepcopy(self._solved), []
        while len(res) != len(self._items):
            next_rank = self._next_rank(satisfied, res)
            satisfied.union(self._mro_satisfied(next_rank))
            res.append(next_rank)
        return res

    def solve(self):
        for key in self.build_dep():
            self._items[key].build_class(key, self._items[key]._loadpath)

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
