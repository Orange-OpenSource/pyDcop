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

from pydcop.replication.path_utils import (
    filter_missing_agents_paths,
    PathsTable,
    head,
    last,
    before_last,
    affordable_path_from,
    remove_path,
)
from pydcop.utils.simple_repr import simple_repr, from_repr


def test_path_head():

    path = ("a1", "a2", "a3")
    assert "a1" == head(path)

    assert head([]) is None


def test_path_iter():

    path = ("a1", "a2", "a3")
    it = iter(path)
    assert "a1" == next(it)
    assert "a2" == next(it)
    assert "a3" == next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_path_last():

    assert "a3" == last(["a1", "a2", "a3"])

    assert last([]) is None


def test_path_before_last():

    assert "a2" == before_last(["a1", "a2", "a3"])

    with pytest.raises(IndexError):
        before_last(["a1"])

    with pytest.raises(IndexError):
        before_last([])


def test_filter_missing_agents_paths():

    paths = [
        (4, ("a2", "a3", "__hosting__")),
        (3, ("a2", "a5", "a6")),
        (1, ("a3", "a4")),
        (3, ("a5", "a3")),
        (3, ("a1", "a4", "a2")),
    ]

    removed = {"a4", "a6"}
    filtered = filter_missing_agents_paths(paths, removed)

    assert len(filtered) == 2


def test_filter_missing_agents_paths_2():
    paths = [
        (1, ("a28", "a16")),
        (1, ("a28", "a22")),
        (1, ("a28", "a32")),
        (1, ("a28", "a35")),
        (1, ("a28", "a02")),
        (1, ("a28", "a20")),
        (1, ("a28", "a06")),
        (1, ("a28", "a03")),
        (1, ("a28", "a26")),
        (1, ("a28", "a14")),
        (1, ("a28", "a24")),
        (11, ("a28", "a18", "__hosting__")),
    ]

    removed = {"a48"}

    filtered = filter_missing_agents_paths(paths, removed)

    assert len(filtered) == len(paths)


@pytest.mark.skip
def test_bench_filter_missing_agents_paths(benchmark):
    def to_bench():
        paths = [
            (1, ("a28", "a16")),
            (1, ("a28", "a22")),
            (1, ("a28", "a32")),
            (1, ("a28", "a35")),
            (1, ("a28", "a02")),
            (1, ("a28", "a20")),
            (1, ("a28", "a06")),
            (1, ("a28", "a03")),
            (1, ("a28", "a26")),
            (1, ("a28", "a14")),
            (1, ("a28", "a24")),
            (11, ("a28", "a18", "__hosting__")),
        ]

        removed = {"a26", "a14"}

        filtered = filter_missing_agents_paths(paths, removed)

        assert len(list(filtered)) == len(paths) - 2

    benchmark(to_bench)
    assert True


def test_path_serialization():

    p = ("a2", "a3", "__hosting__")
    r = simple_repr(p)
    print(r)

    assert "__qualname__" in r
    assert r[0] == "a2"


def test_path_unserialize():
    given = ("a2", "a3", "__hosting__")
    r = simple_repr(given)

    obtained = from_repr(r)
    assert given == obtained


#
#
# def test_pathtable_init():
#     p1 = ("a2", "a3")
#     p2 = ("a2", "a4")
#     table = PathsTable({p1: 2, p2: 3})
#
#     assert len(table) == 2
#
#
# def test_pathtable_get():
#     p1 = ("a2", "a3")
#     table = PathsTable({p1: 2})
#     assert p1 in table
#     assert table[p1] == 2
#
#
# def test_pathtable_iter():
#     p1 = ("a2", "a3")
#     p2 = ("a2", "a4")
#     paths = {p1, p2}
#     table = PathsTable({p1: 2, p2: 3})
#
#     for k in table:
#         paths.remove(k)
#
#     assert len(paths) == 0


def test_remove_path():
    paths = [
        (3, ("a2", "a9", "a4", "a8")),
        (2, ("a2", "a3")),
        (3, ("a2", "a3", "a4")),
        (6, ("a5", "a3", "a4")),
        (4, ("a2", "a3", "a4", "a12")),
        (9, ("a2", "a4", "a4")),
        (3, ("a2", "a4", "a4", "a8")),
        (3, ("a2", "a5", "a4", "a8")),
        (3, ("a2", "a3", "a4", "a8")),
        (3, ("a1", "a3", "a4")),
        (4, ("a2", "a3", "a4", "a1", "a5")),
    ]
    paths.sort()

    remove_path(paths, ("a2", "a3", "a4"))
    assert len(paths) == 10

    remove_path(paths, ("a2", "foo", "a4"))
    assert len(paths) == 10


@pytest.mark.skip
def test_remove_path_bench(benchmark):
    def to_bench():
        paths = [
            (3, ("a2", "a9", "a4", "a8")),
            (2, ("a2", "a3")),
            (3, ("a2", "a3", "a4")),
            (6, ("a5", "a3", "a4")),
            (4, ("a2", "a3", "a4", "a12")),
            (9, ("a2", "a4", "a4")),
            (3, ("a2", "a4", "a4", "a8")),
            (3, ("a2", "a5", "a4", "a8")),
            (3, ("a2", "a3", "a4", "a8")),
            (3, ("a1", "a3", "a4")),
            (4, ("a2", "a3", "a4", "a1", "a5")),
        ]
        paths.sort()

        remove_path(paths, ("a2", "a3", "a4"))
        assert len(paths) == 10

        remove_path(paths, ("a2", "foo", "a4"))
        assert len(paths) == 10

    benchmark(to_bench)
    assert True


def test_pathtable_serialize():
    p1 = ("a2", "a3")
    p2 = ("a2", "a4")
    table = [(2, p1), (3, p2)]

    r = simple_repr(table)
    assert r

    print(r)


def test_pathtable_unserialize():
    p1 = ("a2", "a3")
    p2 = ("a2", "a4")
    table = [(2, p1), (3, p2)]

    r = simple_repr(table)
    assert r

    obtained = from_repr(r)
    assert obtained == table


@pytest.mark.skip
def test_bench_dict_vs_list(benchmark):
    def iterations_dict():
        paths = {
            ("a2", "a9", "a4", "a8"): 3,
            ("a2", "a3"): 2,
            ("a2", "a3", "a4"): 3,
            ("a5", "a3", "a4"): 6,
            ("a2", "a3", "a4", "a12"): 4,
            ("a2", "a4", "a4"): 9,
            ("a2", "a4", "a4", "a8"): 3,
            ("a2", "a5", "a4", "a8"): 3,
            ("a2", "a3", "a4", "a8"): 3,
            ("a1", "a3", "a4"): 3,
            ("a2", "a3", "a4", "a1", "a5"): 4,
        }

        i, foo = 0, None
        for p, c in paths.items():
            i += 1
            foo = (p[:4], c + 1)

    def iterations_list():
        paths = [
            (("a2", "a9", "a4", "a8"), 3),
            (("a2", "a3"), 2),
            (("a2", "a3", "a4"), 3),
            (("a5", "a3", "a4"), 6),
            (("a2", "a3", "a4", "a12"), 4),
            (("a2", "a4", "a4"), 9),
            (("a2", "a4", "a4", "a8"), 3),
            (("a2", "a5", "a4", "a8"), 3),
            (("a2", "a3", "a4", "a8"), 3),
            (("a1", "a3", "a4"), 3),
            (("a2", "a3", "a4", "a1", "a5"), 4),
        ]

        i, foo = 0, None
        for p, c in paths:
            i += 1
            foo = (p[:4], c + 1)

    benchmark(iterations_dict)
    # benchmark(iterations_list)

    assert True


def test_affordable_path_from():
    table = [
        (3, ("a2", "a9", "a4", "a8")),
        (2, ("a2", "a3")),
        (3, ("a2", "a3", "a4")),
        (6, ("a5", "a3", "a4")),
        (4, ("a2", "a3", "a4", "a12")),
        (9, ("a2", "a4", "a4")),
        (3, ("a2", "a4", "a4", "a8")),
        (3, ("a2", "a5", "a4", "a8")),
        (3, ("a2", "a3", "a4", "a8")),
        (3, ("a1", "a3", "a4")),
        (4, ("a2", "a3", "a4", "a1", "a5")),
    ]

    paths = affordable_path_from(("a2", "a3"), 3, table)

    assert len(list(paths)) == 3


@pytest.mark.skip
def test_bench_affordable_path_from(benchmark):
    table = PathsTable(
        {
            ("a2", "a9", "a4", "a8"): 3,
            ("a2", "a3"): 2,
            ("a2", "a3", "a4"): 3,
            ("a5", "a3", "a4"): 6,
            ("a2", "a3", "a4", "a12"): 4,
            ("a2", "a4", "a4"): 9,
            ("a2", "a4", "a4", "a8"): 3,
            ("a2", "a5", "a4", "a8"): 3,
            ("a2", "a3", "a4", "a8"): 3,
            ("a1", "a3", "a4"): 3,
            ("a2", "a3", "a4", "a1", "a5"): 4,
        }
    )

    def to_bench():
        paths = affordable_path_from(("a2", "a3"), 3, table)

        assert len(paths) == 3

    benchmark(to_bench)

    assert True


def test_2():

    roots = ["a2", "a5"]
