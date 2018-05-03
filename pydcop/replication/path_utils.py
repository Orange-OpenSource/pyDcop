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

Node = str


class Path(Sized, Iterable, object):
    """
    Object representing an immutable path.
    """

    def __init__(self, nodes: Union[str, Iterable[Node]]=None, *args) -> None:
        if nodes is None:
            self._path = tuple()  # type: Tuple[Node, ...]
        elif args:
            if isinstance(nodes, str):
                self._path = tuple([nodes] + [str(a) for a in args]) \
                    # type: Tuple[Node, ...]
            else:
                self._path = tuple(list(nodes) + [str(a) for a in args]) \
                    # type: Tuple[Node, ...]
        elif isinstance(nodes, str):
            self._path = (nodes,)  # type: Tuple[Node, ...]
        else:
            self._path = tuple(nodes)  # type: Tuple[Node, ...]

    def head(self) -> Optional[Node]:
        """
        Returns
        -------
        The first element of the path of None if the path is empty
        """
        try:
            return self._path[0]
        except IndexError:
            return None

    def last(self) -> Optional[Node]:
        """

        Returns
        -------
        The last element of the path, or None if the path is empty
        """
        try:
            return self._path[-1]
        except IndexError:
            return None

    def before_last(self):
        """

        Returns
        -------
        The element before the last element in the path

        Raises
        ------
        IndexError if the path has 1 or less elements
        """
        return self._path[-2]

    def tail_if_start_with(self, prefix: 'Path')  \
            -> Optional['Path']:
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
        if self._path[:length] == prefix._path:
            return Path(self._path[length:])
        return None

    @property
    def empty(self):
        """
        Returns
        -------
        True if the path is empty
        """
        return len(self._path) == 0

    def len(self):
        return len(self._path)

    def __len__(self):
        return len(self._path)

    def __iter__(self):
        return iter(self._path)

    def __add__(self, path_extension: 'Path') -> 'Path':
        """ returns a new path built by concatenating this path with
            path_extension"""
        return Path(self._path + path_extension._path)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Path(self._path[key])
        else:
            return self._path[key]

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return self._path == other._path

    def __repr__(self):
        return 'Path({})'.format(self._path)

    def __hash__(self):
        return hash(self._path)


PathsTable = Dict[Path, float]

# TODO : replace Pathstable with specific class
#
# class PathsTable_draft(object):
#
#     def __init__(self):
#         self._paths = {}  # type: Dict[Path, float]
#
#     def filter_invalid(self, agents):
#         self._paths = filter_missing_agents_paths()
#         pass
#
#     def remove(self, path: ReplicationPath_draft) -> Boolean:
#         try:
#             self._paths.pop(path)
#             return True
#         except KeyError:
#             return False
#
#     def add(self, path: ReplicationPath_draft, cost: float):
#         pass
#
#     def cheapest_path_to(self, target: ReplicationComputationName) \
#             -> Tuple(float, ReplicationPath_draft):
#         pass
#

#
#     def path_starting_with(self, prefix: ReplicationPath_draft,
#                        paths: 'PathsTable_draft') -> 'PathsTable_draft':
#         pass


def cheapest_path_to(target: Node,
                     paths: PathsTable)\
        -> Tuple[float, Path]:
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
    c = float('inf')
    for p in paths:
        if p.last() == target:
            return paths[p], p
    return c, Path()


def path_starting_with(prefix: Path,
                       paths: PathsTable) \
        -> List[Tuple[float, Path]]:
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
    found = filter(lambda x: x[1] is not None,
                   ((cost, path.tail_if_start_with(prefix))
                    for path, cost in paths.items()))
    return sorted(found, key=lambda x: x[0])


def affordable_path_from(prefix: Path, max_path_cost: float,
                         paths: PathsTable):
    n_paths = path_starting_with(prefix, paths)
    return {(cost, p)
            for cost, p in n_paths
            if round(cost - max_path_cost, 4) <= 0.0001}


def filter_missing_agents_paths(
        paths: PathsTable,
        replication_computations: Iterable[Node]) \
        -> PathsTable:
    """
    Filters out all paths passing through a replication computation that is not
    available any more.

    Parameters
    ----------
    paths: PathsTable
        known paths with their associated costs
    replication_computations:
        available replication computation names

    Returns
    -------
    A new PathsTable with only valid paths.

    """
    # include the local virtual node in the list of available path : it is
    # not the name of a replication computation but be definitively want to
    # keep it as it is the only node that accepts replicas:
    replication_computations = list(replication_computations) + ['__hosting__']
    filtered = {}
    for path, cost in paths.items():
        missing = [elt for elt in path if elt not in replication_computations]
        if missing:
            continue
        filtered[path] = cost
    return filtered
