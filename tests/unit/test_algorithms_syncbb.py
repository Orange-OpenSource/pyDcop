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
from unittest.mock import MagicMock

import pytest

from pydcop.algorithms import AlgorithmDef, ComputationDef
from pydcop.algorithms.syncbb import (
    get_value_candidates,
    get_next_assignment,
    SyncBBComputation,
    SyncBBForwardMessage,
)
from pydcop.computations_graph.ordered_graph import build_computation_graph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, Variable, create_agents
from pydcop.dcop.relations import constraint_from_str
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.infrastructure.run import solve


def build_pb():
    # A toy problem with 5 variables and 5 constraints.
    # The objective here is to have a problem that is simple enough to be solved
    # manually and used in test, but that is representative enough to be meaningful.

    v_a = Variable("vA", ["R", "G"])
    v_b = Variable("vB", ["R", "G"])
    v_c = Variable("vC", ["R", "G"])
    v_d = Variable("vD", ["R", "G"])
    c1 = constraint_from_str(
        "c1",
        "{('R', 'G'): 8, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 20 "
        "}[(vA, vB)]",
        [v_a, v_b],
    )
    c2 = constraint_from_str(
        "c2",
        "{('R', 'G'): 10, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 20 "
        "}[(vA, vC)]",
        [v_a, v_c],
    )
    c3 = constraint_from_str(
        "c3",
        "{('R', 'G'): 4, "
        " ('R', 'R'): 5, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 3 "
        "}[(vB, vC)]",
        [v_b, v_c],
    )
    c4 = constraint_from_str(
        "c4",
        "{('R', 'G'): 8, "
        " ('R', 'R'): 3, "
        " ('G', 'G'): 3, "
        " ('G', 'R'): 10 "
        "}[(vB, vD)]",
        [v_b, v_d],
    )
    return [v_a, v_b, v_c, v_d], [c1, c2, c3, c4]


@pytest.fixture
def toy_pb():

    return build_pb()


@pytest.fixture
def toy_pb_computation_graph():
    variables, constraints = build_pb()

    # build the pseudo-tree for this problem
    g = build_computation_graph(None, constraints=constraints, variables=variables)
    return g


def get_computation_instance(graph, name):

    # Get the computation node for x1
    comp_node = graph.computation(name)

    # Create the ComputationDef and computation instance
    algo_def = AlgorithmDef.build_with_default_param("syncbb")
    comp_def = ComputationDef(comp_node, algo_def)
    comp = SyncBBComputation(comp_def)
    comp._msg_sender = MagicMock()

    return comp


def test_get_candidates_no_value_selected():
    d = Domain("d", "vals", [0, 1, 2, 3])
    v = Variable("v", d)

    obtained = get_value_candidates(v, None)
    assert obtained == [0, 1, 2, 3]


def test_get_candidate_value_selected():
    d = Domain("d", "vals", ["vB", "vD", "vA", "vE"])
    v = Variable("v", d)

    obtained = get_value_candidates(v, "vB")
    assert obtained == ["vD", "vA", "vE"]

    obtained = get_value_candidates(v, "vA")
    assert obtained == ["vE"]

    obtained = get_value_candidates(v, "vE")
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
        v_b, None, constraints_b, [("vA", "R", 0)], bound, "min"
    )
    assert obtained == ("R", 5)

    constraints_c = [c for c in constraints if v_c in c.dimensions]
    obtained = get_next_assignment(
        v_c, None, constraints_c, [("vA", "R", 0), ("vB", "R", 5)], bound, "min"
    )
    assert obtained == ("R", 10)

    constraints_d = [c for c in constraints if v_d in c.dimensions]
    obtained = get_next_assignment(
        v_d,
        None,
        constraints_d,
        [("vA", "R", 0), ("vB", "R", 5), ("vC", "R", 10)],
        bound,
        "min",
    )
    assert obtained == ("R", 3)


def test_computations_message_at_start(toy_pb_computation_graph):

    # A is the first var in the ordering, it should start selecting a value:
    comp_a = get_computation_instance(toy_pb_computation_graph, "vA")
    assert comp_a.previous_var is None
    assert comp_a.next_var is "vB"
    comp_a.start()
    comp_a._msg_sender.assert_any_call(
        "vA", "vB", SyncBBForwardMessage([("vA", "R", 0)], float("inf")), None, None
    )

    # C is not a start, should not send any message:
    comp_c = get_computation_instance(toy_pb_computation_graph, "vC")
    assert comp_c.previous_var == "vB"
    assert comp_c.next_var == "vD"
    comp_c.start()
    comp_c.message_sender.assert_not_called()


def test_solve_min(toy_pb):

    variables, constraints = toy_pb

    dcop = DCOP(
        name="toy",
        variables={v.name: v for v in variables},
        constraints={c.name: c for c in constraints},
        objective="min"
    )
    dcop.add_agents(create_agents("a", [1, 2, 3, 4]))

    assignment = solve(dcop, "syncbb", "oneagent")

    # Note: this is exactly the same pb as in the file bellow
    # dcop = load_dcop_from_file(["/pyDcop/tests/instances/graph_coloring_tuto.yaml"])

    assert assignment == {"vA": "G", "vB": "G", "vC": "G", "vD": "G"}


def test_solve_max(toy_pb):

    variables, constraints = toy_pb

    dcop = DCOP(
        name="toy",
        variables={v.name: v for v in variables},
        constraints={c.name: c for c in constraints},
        objective="max"
    )
    dcop.add_agents(create_agents("a", [1, 2, 3, 4]))

    assignment = solve(dcop, "syncbb", "oneagent")

    # Note: this is supposed to be exactly the same pb as bellow
    assert assignment == {"vA": "G", "vB": "R", "vC": "R", "vD": "G"}

