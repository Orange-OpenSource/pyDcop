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
from unittest.mock import MagicMock

import pytest

from pydcop.algorithms import ComputationDef, AlgorithmDef
from pydcop.algorithms.ncbb import NcbbAlgo, ValueMessage, CostMessage
from pydcop.computations_graph.pseudotree import PseudoTreeNode, build_computation_graph
from pydcop.dcop.objects import Variable, Domain
from pydcop.dcop.relations import constraint_from_str
from pydcop.infrastructure.computations import ComputationException


@pytest.fixture
def single_variable_pb():
    x1 = Variable("x1", ["R", "B"])
    # build the pseudo-tree for this problem
    g = build_computation_graph(None, constraints=[], variables=[x1])
    return g


@pytest.fixture
def two_variables_pb():
    # a very simple problem with two variables
    x1 = Variable("x1", ["R", "B"])
    x2 = Variable("x2", ["R", "B"])
    diff_x1_x2 = constraint_from_str("c1", "1 if x1 == x2 else 0", [x1, x2])
    # build the pseudo-tree for this problem
    g = build_computation_graph(None, constraints=[diff_x1_x2], variables=[x1, x2])
    return g


@pytest.fixture
def three_variables_pb():
    # a very simple problem with 3 variables
    x1 = Variable("x1", ["R", "B"])
    x2 = Variable("x2", ["R", "B"])
    x3 = Variable("x3", ["R", "B"])
    diff_x1_x2 = constraint_from_str("c1", "1 if x1 == x2 else 0", [x1, x2])
    diff_x1_x3 = constraint_from_str("c2", "1 if x1 == x3 else 0", [x1, x3])
    # build the pseudo-tree for this problem
    g = build_computation_graph(
        None, constraints=[diff_x1_x2, diff_x1_x3], variables=[x1, x2, x3]
    )
    return g


@pytest.fixture
def toy_pb():
    # A toy problem with 5 variables and 5 constraints.
    # The objective here is to have a problem that is simple enough to be solved
    # manually and used in test, but that is representative enough to be meaningful.
    # For example, it includes a loop to make sure we have pseudo parents
    v_a = Variable("A", ["R", "B"])
    v_b = Variable("B", ["R", "B"])
    v_c = Variable("C", ["R", "B"])
    v_d = Variable("D", ["R", "B"])
    v_e = Variable("E", ["R", "B"])
    c1 = constraint_from_str(
        "c1",
        "{('R', 'B'): 1, "
        " ('R', 'R'): 5, "
        " ('B', 'B'): 3, "
        " ('B', 'R'): 2 "
        "}[(A, B)]",
        [v_a, v_b],
    )
    c2 = constraint_from_str(
        "c2",
        "{('R', 'B'): 2, "
        " ('R', 'R'): 8, "
        " ('B', 'B'): 5, "
        " ('B', 'R'): 3 "
        "}[(A, C)]",
        [v_a, v_c],
    )
    c3 = constraint_from_str(
        "c3",
        "{('R', 'B'): 2, "
        " ('R', 'R'): 4, "
        " ('B', 'B'): 2, "
        " ('B', 'R'): 0 "
        "}[(A, D)]",
        [v_a, v_d],
    )
    c4 = constraint_from_str(
        "c4",
        "{('R', 'B'): 0, "
        " ('R', 'R'): 10, "
        " ('B', 'B'): 2, "
        " ('B', 'R'): 1 "
        "}[(B, D)]",
        [v_b, v_d],
    )
    c5 = constraint_from_str(
        "c5",
        "{('R', 'B'): 2, "
        " ('R', 'R'): 4, "
        " ('B', 'B'): 0, "
        " ('B', 'R'): 15 "
        "}[(D, E)]",
        [v_d, v_e],
    )

    # build the pseudo-tree for this problem
    g = build_computation_graph(
        None, constraints=[c1, c2, c3, c4, c5], variables=[v_a, v_b, v_c, v_d, v_e]
    )
    return g


def get_computation_instance(graph, name):

    # Get the computation node for x1
    comp_node = graph.computation(name)

    # Create the ComputationDef and computation instance
    algo_def = AlgorithmDef.build_with_default_param("ncbb")
    comp_def = ComputationDef(comp_node, algo_def)
    comp = NcbbAlgo(comp_def)
    comp._msg_sender = MagicMock()

    return comp


def test_create_computation_no_links(single_variable_pb):

    comp = get_computation_instance(single_variable_pb, "x1")

    assert comp._mode == "min"
    assert comp.is_leaf
    assert comp.is_root
    assert comp.name == "x1"


def test_create_computation_one_neighbor(two_variables_pb):

    # Get the computation instance for x1
    comp = get_computation_instance(two_variables_pb, "x1")

    assert comp._mode == "min"
    assert comp.is_leaf or comp.is_root
    assert comp.name == "x1"


def test_create_computation_three_variables(three_variables_pb):

    # Check computation instance for x1
    comp = get_computation_instance(three_variables_pb, "x1")

    assert comp._mode == "min"
    assert not comp.is_leaf
    assert comp.is_root
    assert comp.name == "x1"
    assert set(comp._children) == {"x2", "x3"}

    # Check computation instance for x2
    comp = get_computation_instance(three_variables_pb, "x2")

    assert comp._mode == "min"
    assert comp.is_leaf
    assert not comp.is_root
    assert comp.name == "x2"
    assert not comp._children
    assert comp._parent == "x1"
    assert comp._ancestors == ["x1"]


def test_problem_with_non_binary_constraints_raises_exception():
    d = Domain("values", "", [1, 0])
    v1 = Variable("v1", d)
    v2 = Variable("v2", d)
    v3 = Variable("v3", d)
    c1 = constraint_from_str("c1", "v1 + v2 + v3 <= 2", [v1, v2, v3])
    g = build_computation_graph(None, constraints=[c1], variables=[v1, v2, v3])
    with pytest.raises(ComputationException) as comp_exc:
        get_computation_instance(g, "v1")
    assert f" with arity {3}" in str(comp_exc.value)

    with pytest.raises(ComputationException) as comp_exc:
        get_computation_instance(g, "v3")
    assert f" with arity {3}" in str(comp_exc.value)

    with pytest.raises(ComputationException) as comp_exc:
        get_computation_instance(g, "v2")
    assert f" with arity {3}" in str(comp_exc.value)


def test_create_computations(toy_pb):
    comp_a = get_computation_instance(toy_pb, "A")

    assert comp_a.is_root
    assert set(comp_a._descendants) == {"D", "B", "C"}

    comp_d = get_computation_instance(toy_pb, "D")
    assert not comp_d.is_root
    assert comp_d._parent == "B"
    assert set(comp_d._ancestors) == {"A", "B"}


def test_select_value_at_root_simple_variable(three_variables_pb):

    comp = get_computation_instance(three_variables_pb, "x1")

    assert comp.current_value is None
    comp.start()

    # When starting the root computation it should select a value
    # and send it to its children.
    assert comp.current_value in ["R", "B"]
    assert comp._msg_sender.call_count == 2

    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg = ValueMessage(comp.current_value)
    msg.cycle_id = 0
    comp._msg_sender.assert_any_call("x1", "x2", msg, None, None)
    comp._msg_sender.assert_any_call("x1", "x3", msg, None, None)


def test_select_value_at_root(toy_pb):

    comp = get_computation_instance(toy_pb, "A")

    assert comp.current_value is None
    comp.start()

    # When starting the root computation it should select a value
    # and send it to its children.
    assert comp.current_value in ["R", "B"]
    assert comp._msg_sender.call_count == 3

    msg = ValueMessage(comp.current_value)
    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg.cycle_id = 0
    comp._msg_sender.assert_any_call("A", "B", msg, None, None)
    comp._msg_sender.assert_any_call("A", "C", msg, None, None)
    comp._msg_sender.assert_any_call("A", "D", msg, None, None)


def test_no_value_selection_at_start_when_not_root(three_variables_pb):

    comp = get_computation_instance(three_variables_pb, "x2")

    assert not comp.is_root
    assert comp.current_value is None
    comp.start()
    comp.message_sender.reset_mock()  # reset startup messages

    # at startup, only the root select a value and send it to its children
    # as x2 is not the root of the pseudo tree, it should not select a variable
    assert comp.current_value is None
    comp.message_sender.assert_not_called()


def test_select_value_in_dfs_only_one_ancestor(toy_pb):

    comp = get_computation_instance(toy_pb, "B")
    comp.start()

    comp.value_phase("A", "R")

    assert comp.current_value == "B"
    assert comp._upper_bound == 1

    msg = ValueMessage("B")
    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg.cycle_id = 0
    comp._msg_sender.assert_any_call("B", "D", msg, None, None)


def test_select_value_in_dfs_two_ancestors(toy_pb):

    comp = get_computation_instance(toy_pb, "D")
    comp.start()

    comp.value_phase("A", "R")
    comp.value_phase("B", "B")

    assert comp.current_value == "B"

    # msg = ValueMessage("B")
    # # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # # the calls will not match, which is quite inconvenient...
    # msg.cycle_id = 0
    # comp._msg_sender.assert_any_call("B", "D", msg, None, None)


def test_cost_msg_from_leaf(toy_pb):

    comp_c = get_computation_instance(toy_pb, "C")
    comp_c.start()

    comp_c.value_phase("A", "R")

    assert comp_c.current_value == "B"

    msg = CostMessage(2)
    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg.cycle_id = 0
    comp_c._msg_sender.assert_any_call("C", "A", msg, None, None)


def test_cost_msg_from_subtree_d(toy_pb):

    comp_d = get_computation_instance(toy_pb, "D")
    comp_d.start()

    comp_d._upper_bound = 2
    comp_d.cost_phase("E", 0)

    assert comp_d._upper_bound == 2

    msg = CostMessage(2)
    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg.cycle_id = 0
    comp_d._msg_sender.assert_any_call("D", "B", msg, None, None)


def test_cost_msg_from_subtree_b(toy_pb):

    comp_b = get_computation_instance(toy_pb, "B")
    comp_b.start()
    comp_b._upper_bound = 1
    comp_b.cost_phase("D", 2)

    assert comp_b._upper_bound == 3

    msg = CostMessage(3)
    # Warning, the messages that are sent contains the cycle_id, if we don't add them
    # the calls will not match, which is quite inconvenient...
    msg.cycle_id = 0
    comp_b._msg_sender.assert_any_call("B", "A", msg, None, None)


def test_cost_msg_at_root(toy_pb):

    comp_a = get_computation_instance(toy_pb, "A")
    comp_a.start()
    comp_a._upper_bound = 0
    comp_a.cost_phase("B", 3)
    comp_a.cost_phase("C", 0)

    assert comp_a._upper_bound == 3
