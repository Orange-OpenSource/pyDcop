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


from typing import Iterable, Optional, Tuple, List, Set

Node = str
Path = Tuple[Node, ...]


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


PathsTable = List[Tuple[float, Path]]


def remove_path(paths: PathsTable, path: Path) -> PathsTable:
    """
     Remove a path from a list of paths. Maintains ordering

    Parameters
    ----------
    paths
    path

    Returns
    -------

    """
    to_remove = [(c, p) for c, p in paths if p == path]
    for item in to_remove:
        paths.remove(item)
    return paths


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
    for cost, p in paths:
        if p[-1] == target:
            return cost, p
    return float("inf"), ()


def affordable_path_from(prefix: Path, max_path_cost: float, paths: PathsTable):
    # filtered = []
    plen = len(prefix)
    for cost, path in paths:
        if path[:plen] == prefix and (cost - max_path_cost) <= 0.0001:
            yield path[plen:]
            # filtered.append((cost, path[plen:]))
    # return filtered


def filter_missing_agents_paths(
    paths: PathsTable, removed_agents: Set
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
    filtered = []
    for cost, path in paths:
        missing = False
        for elt in path:
            if elt in removed_agents:
                missing = True
                break
        if missing:
            continue
        filtered.append((cost, path))
    return filtered
