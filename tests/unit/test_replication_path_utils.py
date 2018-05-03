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


import pytest

from pydcop.replication.path_utils import Path, path_starting_with, \
    filter_missing_agents_paths


def test_path_creation():

    assert len(Path(['a', 'b'])) == 2
    assert len(Path(['a1'])) == 1
    assert len(Path('a1')) == 1
    assert len(Path('a', 'b', 'c')) == 3
    assert len(Path('a1', 'a2', 'a3')) == 3
    assert len(Path(['a1', 'a2'], 'a3')) == 3
    assert len(Path()) == 0


def test_path_head():

    path = Path('a1', 'a2', 'a3')
    assert 'a1' == path.head()

    path = Path([])
    assert path.head() is None


def test_path_iter():

    path = Path('a1', 'a2', 'a3')
    it = iter(path)
    assert 'a1' == next(it)
    assert 'a2' == next(it)
    assert 'a3' == next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_path_item():

    path = Path('a1', 'a2', 'a3')
    assert 'a1' == path[0]
    assert Path('a1', 'a2') == path[:-1]


def test_path_empty():
    path = Path()
    assert path.empty

    path = Path('a1', 'a2', 'a3')
    assert not path.empty


def test_path_last():

    path = Path(['a1', 'a2', 'a3'])
    assert 'a3' == path.last()

    path = Path([])
    assert path.last() is None


def test_path_before_last():

    path = Path(['a1', 'a2', 'a3'])
    assert 'a2' == path.before_last()

    path = Path(['a1'])
    with pytest.raises(IndexError):
        path.before_last()

    path = Path([])
    with pytest.raises(IndexError):
        path.before_last()


def test_path_add():

    p1 = Path('a1')
    p2 = Path('a2', 'a3')
    assert p1 + p2 == Path('a1', 'a2', 'a3')
    assert p2 + p1 == Path('a2', 'a3', 'a1')

    p1 = Path()
    p2 = Path('a2', 'a3')
    assert p1 + p2 == Path('a2', 'a3')
    assert p2 + p1 == Path('a2', 'a3')

    p1 = Path([])
    p2 = Path([])
    assert p1 + p2 == Path()
    assert p2 + p1 == Path()


def test_path_tail_if_start_with():
    path = Path(('A', 'B'))
    obtained = path.tail_if_start_with(Path('A'))
    assert obtained == Path(('B',))

    obtained = path.tail_if_start_with(Path('A', 'B'))
    assert obtained == Path(())

    obtained = Path('A', 'B', 'C', 'D')\
        .tail_if_start_with(Path('A', 'B'))
    assert obtained == Path('C', 'D')

    obtained = Path('A', 'B', 'C', 'D')\
        .tail_if_start_with(Path('A', 'D'))
    assert obtained is None

    obtained = Path(()).tail_if_start_with(Path('A', 'B'))
    assert obtained is None

    obtained = Path(('A', 'B', 'C', 'D'))\
        .tail_if_start_with(Path())
    assert obtained == Path(('A', 'B', 'C', 'D'))


def test_paths_starting_with():
    paths_starting_a2 = path_starting_with(
        Path('_replication_a2', ),
        {Path('_replication_a2', '_replication_a3'): 4,
         Path('_replication_a2', '_replication_a5', '_replication_a6'): 3,
         Path('_replication_a3', '_replication_a4'): 1,
         Path('_replication_a4', '_replication_a3'): 3,
         })

    cost, path = paths_starting_a2[0]
    assert cost == 3
    assert path == Path('_replication_a5', '_replication_a6')
    paths = [path for _, path in paths_starting_a2]
    assert Path('_replication_a3', ) in paths
    assert Path('_replication_a5', '_replication_a6') in paths


def test_filter_missing_agents_paths():

    paths = {
        Path('_replication_a2', '_replication_a3', '__hosting__'): 4,
        Path('_replication_a2', '_replication_a5', '_replication_a6'): 3,
        Path('_replication_a3', '_replication_a4'): 1,
        Path('_replication_a5', '_replication_a3'): 3,
        Path('_replication_a1', '_replication_a4', '_replication_a2'): 3,
    }

    available = {'_replication_a2', '_replication_a3',
                 '_replication_a5', '_replication_a6'}
    filtered = filter_missing_agents_paths(paths, available)

    assert len(filtered) == 3
