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

"""
Unit tests for the SyncBB algorithm.

"""
import pytest

from pydcop.algorithms.syncbb import get_value_candidates, get_next_assignment
from pydcop.computations_graph.pseudotree import build_computation_graph
from pydcop.dcop.objects import Domain, Variable
from pydcop.dcop.relations import constraint_from_str


@pytest.fixture
def toy_pb():
    # A toy problem with 5 variables and 5 constraints.
    # The objective here is to have a problem that is simple enough to be solved
    # manually and used in test, but that is representative enough to be meaningful.
    v_a = Variable("A", ["R", "G"])
    v_b = Variable("B", ["R", "G"])
    v_c = Variable("C", ["R", "G"])
    v_d = Variable("D", ["R", "G"])
    c1 = constraint_from_str(
        "c1",
        "{('R', 'G'): 8, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 20 "
        "}[(A, B)]",
        [v_a, v_b],
    )
    c2 = constraint_from_str(
        "c2",
        "{('R', 'G'): 10, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 20 "
        "}[(A, C)]",
        [v_a, v_c],
    )
    c3 = constraint_from_str(
        "c3",
        "{('R', 'G'): 4, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 3 "
        "}[(B, C)]",
        [v_b, v_c],
    )
    c4 = constraint_from_str(
        "c4",
        "{('R', 'G'): 8, "
        " ('R', 'R'): 3, "
        " ('G', 'G'): 8, "
        " ('G', 'R'): 10 "
        "}[(B, D)]",
        [v_b, v_d],
    )
    return [v_a, v_b, v_c, v_d], [c1, c2, c3, c4]



def test_get_candidates_no_value_selected():
    d = Domain("d", "vals", [0, 1, 2, 3])
    v = Variable("v", d)

    obtained = get_value_candidates(v, None)
    assert obtained == [0, 1, 2, 3]


def test_get_candidate_value_selected():
    d = Domain("d", "vals", ["B", "D", "A", "E"])
    v = Variable("v", d)

    obtained = get_value_candidates(v, "B")
    assert obtained == ["D", "A", "E"]

    obtained = get_value_candidates(v, "A")
    assert obtained == ["E"]

    obtained = get_value_candidates(v, "E")
    assert obtained == []


def test_get_next_assignement_empty_path_no_bound(toy_pb):

    variables, constraints = toy_pb
    variable = variables[0]
    var_constraints = [c for c in constraints if variable in c.dimensions]
    bound = float("inf")

    obtained = get_next_assignment(variable, None, var_constraints, [], bound, "min")
    assert obtained == ("R", 0)


def test_get_next_assignment_no_bound(toy_pb):

    variables, constraints = toy_pb
    v_a, v_b, v_c, v_d = variables
    bound = float("inf")

    constraints_b = [c for c in constraints if v_b in c.dimensions]
    obtained = get_next_assignment(
        v_b, None, constraints_b, [("A", "R", 0)], bound, "min"
    )
    assert obtained == ("R", 5)

    constraints_c = [c for c in constraints if v_c in c.dimensions]
    obtained = get_next_assignment(
        v_c, None, constraints_c, [("A", "R", 0), ("B", "R", 5)], bound, "min"
    )
    assert obtained == ("R", 10)

    constraints_d = [c for c in constraints if v_d in c.dimensions]
    obtained = get_next_assignment(
        v_d,
        None,
        constraints_d,
        [("A", "R", 0), ("B", "R", 5), ("C", "R", 10)],
        bound,
        "min",
    )
    assert obtained == ("R", 3)
