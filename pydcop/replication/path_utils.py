# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from typing import Iterable, Optional, Dict, Tuple, List, Sized, Union

from pydcop.utils.simple_repr import SimpleRepr, simple_repr, from_repr

Node = str
Path = Tuple[Node, ...]


def tail_if_start_with(path: Tuple, prefix: Tuple) -> Optional["Path"]:
    """
    Checks if starts with `prefix` and returns its tail in that case.

    Parameters
    ----------
    prefix: Replication path
        a path that might be a prefix if `path`

    Returns
    -------
    an optional replication path
        the fail of `path` (everything after the prefix) if it starts with
        `prefix`, None otherwise.
    """
    length = len(prefix)
    if path[:length] == prefix:
        return path[length:]
    return None


def head(path) -> Optional[Node]:
    """
    Returns
    -------
    The first element of the path of None if the path is empty
    """
    try:
        return path[0]
    except IndexError:
        return None


def last(path) -> Optional[Node]:
    """

    Returns
    -------
    The last element of the path, or None if the path is empty
    """
    try:
        return path[-1]
    except IndexError:
        return None


def before_last(path):
    """

    Returns
    -------
    The element before the last element in the path

    Raises
    ------
    IndexError if the path has 1 or less elements
    """
    return path[-2]


# PathsTable = Dict[Path, float]


#PathsTable = List[Tuple[Path, float]]

class PathsTable(SimpleRepr):
    """
    A PathsTable associate a Path to a cost.

    """

    def __init__(self, table: Dict[Path, float] = None):
        if table:
            self.table = table
        else:
            self.table: Dict[Path, float] = {}

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        self.table.__setitem__(key, value)

    def __contains__(self, key):
        return key in self.table

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        return iter(self.table)

    def __eq__(self, other):
        return self.table == other.table

    def items(self):
        return self.table.items()

    def values(self):
        return self.table.values()

    def pop(self, key):
        return self.table.pop(key)

    def copy(self):
        return self.table.copy()

    def __repr__(self):
        # FIXME : optim perf
        return "Pathtable"
        # return repr(self.table)

    def _simple_repr(self):

        # Full name = module + qualifiedname (for inner classes)
        r = {"__module__": self.__module__, "__qualname__": self.__class__.__qualname__}
        content = simple_repr(list(self.table.items()))
        r["paths"] = content

        return r

    @classmethod
    def _from_repr(cls, r):

        # assert r["__module__"] == self.__module__
        # assert r['__qualname__'] == self.__class__.__qualname__

        table = {}
        for path in r["paths"]:
            p, c = from_repr(path)
            table[p] = c

        return PathsTable(table)


def cheapest_path_to(target: Node, paths: PathsTable) -> Tuple[float, Path]:
    """
    Search the cheapest path and its costs in `paths` that ends at `target`.

    Parameters
    ----------
    target: Node
        The end node to look for
    paths: dict of path, float
        A dict of known paths with their costs

    Returns
    -------
    Tuple[float, Path]
        A Tuple containing the cheapest cost and the corresponding path. If
        `paths` contains no path ending at target, return an infinite cost with
        an empty path.

    :return:
    """
    c = float("inf")
    for p, cost in paths.items():
        if p[-1] == target:
            return cost, p
    return c, ()


def path_starting_with(prefix: Path, paths: PathsTable) -> List[Tuple[float, Path]]:
    """
    Search in `paths` all of paths starting with the prefix `start_path`,
    and return them as a list (excluding the prefix) in increasing cost order.

    Parameters
    ----------
    prefix: Path
        path prefix to look for
    paths: PathsTable (dict of Path float)
        a dict of known paths with their costs

    Returns
    -------
    List[Tuple[float, Path]]
        a list of tuple (cost, path_without_prefix)
    """
    # found = filter(
    #     lambda x: x[1] is not None,
    #     ((cost, path.tail_if_start_with(prefix)) for path, cost in paths.items()),
    # )
    # return sorted(found, key=lambda x: x[0])

    # Previous, slower, implementation:
    # tails = ((cost, tail_if_start_with(path, prefix)) for path, cost in paths.items())
    # return sorted(((cost, tail) for cost, tail in tails if tail is not None), key=lambda x: x[0])

    filtered = []
    plen = len(prefix)
    for path, cost in paths.items():
        if path[:plen] == prefix:
            filtered.append((cost, path[plen:]))
    filtered.sort()
    return filtered


def affordable_path_from(prefix: Path, max_path_cost: float, paths: PathsTable):
    filtered = []
    plen = len(prefix)
    for path, cost in paths.items():
        if path[:plen] == prefix and (cost - max_path_cost) <= 0.0001:
            filtered.append((cost, path[plen:]))
    filtered.sort()
    return filtered

    # n_paths = path_starting_with(prefix, paths)
    # return {
    #     (cost, p) for cost, p in n_paths if round(cost - max_path_cost, 4) <= 0.0001
    # }


def filter_missing_agents_paths(
    paths: PathsTable, removed_agents: Iterable[Node]
) -> PathsTable:
    """
    Filters out all paths passing through an agent that is not
    available any more.

    Parameters
    ----------
    paths: PathsTable
        known paths with their associated costs
    removed_agents:
        names of removed agents

    Returns
    -------
    A new PathsTable with only valid paths.

    """
    # include the local virtual node in the list of available path : it is
    # not the name of a replication computation but be definitively want to
    # keep it as it is the only node that accepts replicas:

    # Two attempts of making it faster: no success
    # return {path: cost for path, cost in paths.items()
    #         if all(elt not in removed_agents for elt in path)}

    # return filter(lambda p : all(elt not in removed_agents for elt in p[0]), paths.items())

    filtered = {}
    for path, cost in paths.items():
        missing = False
        for elt in path:
            if elt in removed_agents:
                missing = True
                break
        if missing:
            continue
        filtered[path] = cost
    return PathsTable(filtered)
