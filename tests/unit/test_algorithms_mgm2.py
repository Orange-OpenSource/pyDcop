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


import unittest
from unittest.mock import MagicMock

import pytest

from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    UnaryFunctionRelation,
    AsNAryFunctionRelation,
    constraint_from_str,
)

from pydcop.algorithms import mgm2, AlgorithmDef, ComputationDef
from pydcop.algorithms.mgm2 import (
    Mgm2Computation,
    Mgm2ValueMessage,
    Mgm2OfferMessage,
    Mgm2GainMessage,
    Mgm2ResponseMessage,
    Mgm2GoMessage,
)
from pydcop.dcop.objects import Variable
from tests.unit.test_algorithms_dpop import DummySender


def test_communication_load():
    v1 = Variable("v1", list(range(10)))
    v2 = Variable("v2", list(range(10)))
    v3 = Variable("v3", list(range(10)))
    v4 = Variable("v4", list(range(10)))
    c1 = constraint_from_str("c1", " v1 == v2", [v1, v2])
    c2 = constraint_from_str("c2", " v1 == v3", [v1, v3])
    c3 = constraint_from_str("c3", " v1 == v4", [v1, v4])
    v1_node = VariableComputationNode(v1, [c1, c2, c3])

    assert mgm2.UNIT_SIZE * 10 * 10 * 3 + mgm2.HEADER_SIZE == mgm2.communication_load(
        v1_node, "v2"
    )


def test_computation_memory_one_constraint():
    v1 = Variable("v1", list(range(10)))
    v2 = Variable("v2", list(range(10)))
    v3 = Variable("v3", list(range(10)))
    c1 = constraint_from_str("c1", " v1 + v2 == v3", [v1, v2, v3])
    v1_node = VariableComputationNode(v1, [c1])

    # here, we have an hyper-edges with 3 vertices
    assert mgm2.computation_memory(v1_node) == mgm2.UNIT_SIZE * 2 * 2


def test_computation_memory_two_constraints():
    v1 = Variable("v1", list(range(10)))
    v2 = Variable("v2", list(range(10)))
    v3 = Variable("v3", list(range(10)))
    v4 = Variable("v4", list(range(10)))
    c1 = constraint_from_str("c1", " v1 == v2", [v1, v2])
    c2 = constraint_from_str("c2", " v1 == v3", [v1, v3])
    c3 = constraint_from_str("c3", " v1 == v4", [v1, v4])
    v1_node = VariableComputationNode(v1, [c1, c2, c3])

    # here, we have 3 edges , one for each constraint
    assert mgm2.computation_memory(v1_node) == mgm2.UNIT_SIZE * 3 * 2


def test_no_neighbors():
    x1 = Variable("x1", list(range(10)))
    cost_x1 = constraint_from_str("cost_x1", "x1 *2 ", [x1])

    computation = Mgm2Computation(
        ComputationDef(
            VariableComputationNode(x1, [cost_x1]),
            AlgorithmDef.build_with_default_param("mgm2", mode="max"),
        )
    )

    computation.value_selection = MagicMock()
    computation.finished = MagicMock()
    vals, cost = computation._compute_best_value()
    assert cost == 18
    assert set(vals) == {9}

    computation.on_start()
    computation.value_selection.assert_called_once_with(9, 18)
    computation.finished.assert_called_once_with()


class TestsValueComputation(unittest.TestCase):
    def test_best_unary(self):
        x = Variable("x", list(range(5)))
        phi = UnaryFunctionRelation("phi", x, lambda x_: 1 if x_ in [0, 2, 3] else 0)

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.__value__ = 0
        bests, best = computation._compute_best_value()

        self.assertEqual(best, 0)
        self.assertEqual(bests, [1, 4])

    def test_binary_func_min(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )

        computation._neighbors_values["x2"] = 1
        bests, best = computation._compute_best_value()

        self.assertEqual(bests, [0])
        self.assertEqual(best, 1)

    def test_binary_func_max(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation._neighbors_values["x2"] = 1
        bests, best = computation._compute_best_value()

        self.assertEqual(bests, [1])
        self.assertEqual(best, 2)

    def test_3_ary_func_min(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", [1])

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation._neighbors_values["x2"] = 1
        computation._neighbors_values["x3"] = 1
        bests, best = computation._compute_best_value()

        self.assertEqual(bests, [0])
        self.assertEqual(best, 2)

    def test_3_ary_func_max(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", [1])

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation._neighbors_values["x2"] = 1
        computation._neighbors_values["x3"] = 1
        bests, best = computation._compute_best_value()

        self.assertEqual(bests, [1])
        self.assertEqual(best, 3)


class TestsCostComputation(unittest.TestCase):
    def test_unary_function_relation(self):
        x = Variable("x", list(range(5)))
        #        x2 = Variable('x2', list(range(5)))

        #        @AsNAryFunctionRelation(x, x2)
        #       def phi(x1_):
        #          return x1_
        phi = UnaryFunctionRelation("phi", x, lambda x_: 1 if x_ in [0, 2, 3] else 0)
        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.__value__ = 0

        self.assertEqual(computation._compute_cost(**{"x": 0}), 1)

    def test_binary_func(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        self.assertEqual(computation._compute_cost(**{"x1": 0, "x2": 0}), 0)
        self.assertEqual(computation._compute_cost(**{"x1": 0, "x2": 1}), 1)
        self.assertEqual(computation._compute_cost(**{"x1": 1, "x2": 0}), 1)
        self.assertEqual(computation._compute_cost(**{"x1": 1, "x2": 1}), 2)

    def test_3_ary_func(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", [1])

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        self.assertEqual(computation._compute_cost(**{"x1": 0, "x2": 0, "x3": 1}), 1)
        self.assertEqual(computation._compute_cost(**{"x1": 0, "x2": 1, "x3": 1}), 2)
        self.assertEqual(computation._compute_cost(**{"x1": 1, "x2": 0, "x3": 1}), 2)
        self.assertEqual(computation._compute_cost(**{"x1": 1, "x2": 1, "x3": 1}), 3)

    def test_current_local_cost_unary(self):
        x = Variable("x", list(range(5)))
        #        x2 = Variable('x2', list(range(5)))

        #        @AsNAryFunctionRelation(x, x2)
        #       def phi(x1_):
        #          return x1_
        phi = UnaryFunctionRelation("phi", x, lambda x_: 1 if x_ in [0, 2, 3] else 0)
        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.__value__ = 0
        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation2.__value__ = 1

        self.assertEqual(computation._current_local_cost(), 1)
        self.assertEqual(computation2._current_local_cost(), 0)

    def test_current_local_cost_binary(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.__value__ = 1
        computation._neighbors_values["x2"] = 0

        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation2.__value__ = 0
        computation2._neighbors_values["x2"] = 0
        self.assertEqual(computation._current_local_cost(), 1)
        self.assertEqual(computation2._current_local_cost(), 0)

    def test_current_local_cost_3_ary(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", [1])

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.__value__ = 1
        computation._neighbors_values["x2"] = 0
        computation._neighbors_values["x3"] = 1

        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation2.__value__ = 0
        computation2._neighbors_values["x2"] = 0
        computation2._neighbors_values["x3"] = 1
        self.assertEqual(computation._current_local_cost(), 2)
        self.assertEqual(computation2._current_local_cost(), 1)


class TestsChangeState(unittest.TestCase):
    def test_enter_value_state(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation.message_sender = DummySender()
        computation.__value__ = 1

        computation._postponed_msg["value"] = [("x2", Mgm2ValueMessage(5), 1)]

        computation._enter_state("value")

        self.assertEqual(computation._state, "value")
        self.assertEqual(computation._postponed_msg["value"], [])
        self.assertEqual(computation._neighbors_values["x2"], 5)

    def test_enter_offer_state(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation.message_sender = DummySender()

        computation._postponed_msg["offer"] = [
            ("x2", Mgm2OfferMessage({(1, 1): 5}, is_offering=True), 1)
        ]

        computation._enter_state("offer")

        self.assertEqual(computation._state, "offer")
        self.assertEqual(computation._postponed_msg["offer"], [])
        self.assertEqual(
            computation._offers, [("x2", Mgm2OfferMessage({(1, 1): 5}, True))]
        )

    def test_enter_answer_state(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation.message_sender = DummySender()

        computation._enter_state("answer?")

        self.assertEqual(computation._state, "answer?")

    def test_enter_gain_state(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation.message_sender = DummySender()

        computation._postponed_msg["gain"] = [("x2", Mgm2GainMessage(3), 1)]

        computation._enter_state("gain")

        self.assertEqual(computation._state, "gain")
        self.assertEqual(computation._postponed_msg["gain"], [])
        self.assertEqual(computation._neighbors_gains["x2"], 3)

    def test_enter_go_state(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation.message_sender = DummySender()

        computation._enter_state("go?")

        self.assertEqual(computation._state, "go?")


class TestsOffersComputations(unittest.TestCase):
    def test_compute_offers_min_mode(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 0}
        computation._partner = x2
        computation.__value__ = 0
        computation.__cost__ = 2
        offers = computation._compute_offers_to_send()

        self.assertEqual(offers, {(1, 0): 2, (1, 1): 1})

    def test_compute_offers_max_mode(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 0}
        computation._partner = x2
        computation.__value__ = 1
        computation.__cost__ = 0
        offers = computation._compute_offers_to_send()

        self.assertEqual(offers, {(0, 0): -2, (0, 1): -2, (1, 1): -1})

    def test_find_best_offer_min_mode_one_offerer(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))
        x4 = Variable("x4", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x4)
        def psi(x1_, x4_):
            if x1_ == x4_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 0, "x4": 0}
        computation.__value__ = 0
        computation.__cost__ = 3

        bests, best_gain = computation._find_best_offer(
            [("x2", {(0, 0): 1, (0, 1): 5, (1, 0): 3})]
        )
        bests2, best_gain2 = computation._find_best_offer(
            [("x2", {(0, 0): 1, (0, 1): 5, (1, 0): 6})]
        )

        self.assertEqual(bests, [(0, 1, "x2")])
        self.assertEqual(best_gain, 8)
        self.assertEqual(set(bests2), {(0, 1, "x2"), (1, 0, "x2")})
        self.assertEqual(best_gain2, 8)

    def test_find_best_offer_max_mode_one_offerer(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))
        x4 = Variable("x4", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x4)
        def psi(x1_, x4_):
            if x1_ == x4_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 1, "x4": 1}
        computation.__value__ = 0
        computation.__cost__ = 1

        bests, best_gain = computation._find_best_offer(
            [("x2", {(0, 0): -1, (0, 1): -5, (1, 0): -3})]
        )
        # global gain: -1 -5 -5

        bests2, best_gain2 = computation._find_best_offer(
            [("x2", {(0, 0): -1, (0, 1): -5, (1, 0): -6})]
        )
        # global gain: -1 -5 -5

        self.assertEqual(bests, [(0, 1, "x2")])
        self.assertEqual(best_gain, -5)
        self.assertEqual(set(bests2), {(0, 1, "x2"), (1, 0, "x2")})
        self.assertEqual(best_gain2, -5)

    def test_find_best_offer_min_mode_2_offerers(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))
        x4 = Variable("x4", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x4)
        def psi(x1_, x4_):
            if x1_ == x4_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 0, "x4": 0}
        computation.__value__ = 0
        computation.__cost__ = 3

        bests, best_gain = computation._find_best_offer(
            [
                ("x2", {(0, 0): 1, (0, 1): 5, (1, 0): 3}),
                ("x4", {(1, 0): 7, (0, 1): 2, (1, 1): 3}),
            ]
        )

        self.assertEqual(set(bests), {(0, 1, "x2"), (1, 0, "x4")})
        self.assertEqual(best_gain, 8)

    def test_find_best_offer_max_mode_2_offerers(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))
        x4 = Variable("x4", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x3_:
                return 2
            elif x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x4)
        def psi(x1_, x4_):
            if x1_ == x4_:
                return 1
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )

        computation._neighbors_values = {"x2": 0, "x3": 1, "x4": 1}
        computation._partner = x2
        computation.__value__ = 0
        computation.__cost__ = 1

        bests, best_gain = computation._find_best_offer(
            [
                ("x2", {(0, 0): -1, (0, 1): -5, (1, 0): -3}),
                ("x4", {(1, 0): -5, (0, 1): -4, (1, 1): -3}),
            ]
        )

        self.assertEqual(set(bests), {(0, 1, "x2"), (0, 1, "x4"), (1, 0, "x4")})
        self.assertEqual(best_gain, -5)


class TestsHandleMessage(unittest.TestCase):
    def test_value_not_all_neighbors_received(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation._state = "value"
        computation.on_value_msg("x2", Mgm2ValueMessage(0), 1)

        self.assertEqual(computation._state, "value")
        self.assertEqual(computation._neighbors_values["x2"], 0)

    def test_value_all_neighbors_received(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._state = "value"
        computation.__value__ = 1
        computation.on_value_msg("x2", Mgm2ValueMessage(0), 1)

        self.assertEqual(computation._state, "offer")
        self.assertEqual(computation._neighbors_values["x2"], 0)
        self.assertEqual(computation._potential_gain, 1)
        self.assertEqual(computation._potential_value, 0)

        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2", mode="max"),
            )
        )
        computation2.message_sender = DummySender()
        computation2._state = "value"
        computation2.__value__ = 1
        computation2.on_value_msg("x2", Mgm2ValueMessage(0), 1)
        self.assertEqual(computation2._state, "offer")
        self.assertEqual(computation2._neighbors_values["x2"], 0)
        self.assertEqual(computation2._potential_gain, 0)
        self.assertEqual(computation2._potential_value, 1)

    @pytest.mark.skip
    def test_offer_has_no_partner_yet(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        # Receives a fake offer
        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._state = "offer"
        computation.on_offer_msg("x2", Mgm2OfferMessage(), 1)
        self.assertEqual(computation._state, "offer")
        self.assertEqual(computation._offers, [])
        # Received only fake offers
        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation2.message_sender = DummySender()
        computation2._state = "offer"
        computation2.__nb_received_offers__ = 1
        computation2.on_offer_msg("x2", Mgm2OfferMessage(), 1)
        self.assertEqual(computation2._state, "gain")
        self.assertEqual(computation2._offers, [])

        # Receives a real offer (but still expects other OfferMessages)
        computation3 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation3.message_sender = DummySender()
        computation3._state = "offer"
        computation3.on_offer_msg(
            "x2", Mgm2OfferMessage({(1, 1): 8}, is_offering=True), 1
        )
        self.assertEqual(computation3._state, "offer")
        self.assertEqual(computation3._offers, [("x2", {(1, 1): 8})])
        # Receives a real offer and is the last expected OfferMessage
        computation4 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation4.message_sender = DummySender()
        computation4._state = "offer"
        computation4._neighbors_values = {"x2": 0, "x3": 1}
        computation4.__value__ = 0
        computation4.__cost__ = 1
        computation4.__nb_received_offers__ = 1
        computation4.on_offer_msg(
            "x2", Mgm2OfferMessage({(1, 1): 8}, is_offering=True), 1
        )
        self.assertEqual(computation4._offers, [("x2", {(1, 1): 8})])
        self.assertEqual(computation4._state, "gain")
        self.assertEqual(computation4._potential_gain, 9)
        self.assertEqual(computation4._potential_value, 1)

    @pytest.mark.skip
    def test_offer_already_has_partner(self):
        x1 = Variable("x1", list(range(2)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            return x1_ + x2_ + x3_

        # Receives a fake offer
        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._state = "offer"
        computation._is_offerer = True
        computation.on_offer_msg("x2", Mgm2OfferMessage(), 1)
        self.assertEqual(computation._state, "offer")
        # self.assertEqual(computation._offers, [])
        # Received only fake offers
        computation2 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation2.message_sender = DummySender()
        computation2._state = "offer"
        computation2.__nb_received_offers__ = 1
        computation2.on_offer_msg("x2", Mgm2OfferMessage(), 1)
        self.assertEqual(computation2._state, "gain")
        self.assertEqual(computation2._offers, [("x2", Mgm2OfferMessage())])
        # receives a real offer
        computation3 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation3.message_sender = DummySender()
        computation3._state = "offer"
        computation3._is_offerer = True
        computation3.__cost__ = 15
        computation3.on_offer_msg(
            "x2", Mgm2OfferMessage({(1, 1): 8}, is_offering=True), 1
        )
        self.assertEqual(computation3._state, "offer")
        self.assertEqual(2, len(computation3._offers))
        self.assertEqual(computation3._potential_gain, 0)
        self.assertIsNone(computation3._potential_value)
        # Receives a real offer which is the last expected one
        computation4 = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation4.message_sender = DummySender()
        computation4._state = "offer"
        computation4._is_offerer = True
        computation4.__nb_received_offers__ = 1
        computation4.on_offer_msg(
            "x2", Mgm2OfferMessage({(1, 1): 8}, is_offering=True), 1
        )
        self.assertEqual(len(computation4), 3)
        self.assertEqual(computation4._state, "answer?")
        self.assertEqual(computation4._potential_gain, 0)
        self.assertIsNone(computation4._potential_value)

    #
    # def test_offer_has_better_unilateral_move(self):
    #     x1 = Variable("x1", list(range(2)))
    #     x2 = Variable('x2', list(range(2)))
    #     x3 = Variable('x3', list(range(2)))
    #
    #     @AsNAryFunctionRelation(x1, x2)
    #     def phi(x1_, x2_):
    #         if x1_ == x2_:
    #             return 1
    #         return 0
    #
    #     @AsNAryFunctionRelation(x1, x3)
    #     def psi(x1_, x3_):
    #         if x1_ == x3_:
    #             return 8
    #         return 0
    #
    #     # Receives a real offer from last neighbor
    #     computation = Mgm2Computation(
    #         ComputationDef(
    #             VariableComputationNode(x1, [phi, psi]),
    #             AlgorithmDef.build_with_default_param('mgm2')
    #         ))
    #     computation.message_sender = DummySender()
    #
    #     computation._state = 'offer'
    #     computation._neighbors_values = {'x2': 1, 'x3': 1}
    #     computation.__value__ = 1
    #     computation.__cost__ = 9
    #     computation._potential_gain = 9  # best unilateral move
    #     computation._potential_value = 0  # best unilateral move
    #     computation.__nb_received_offers__ = 1
    #     computation.on_offer_msg('x2', Mgm2OfferMessage({(0, 1): 1},
    #                                                          is_offering=True), 1)
    #     self.assertEqual(computation._offers, [('x2', Mgm2OfferMessage({(0, 1): 1},
    #                                                          is_offering=True))])
    #     self.assertEqual(computation._state, 'gain')
    #     self.assertEqual(computation._potential_gain, 9)
    #     self.assertEqual(computation._potential_value, 0)

    def test_response_accept(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._state = "answer?"
        computation._is_offerer = True
        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation.__cost__ = 9
        computation._potential_gain = 9  # best unilateral move
        computation._potential_value = 2  # best unilateral move
        computation._partner = x3

        computation.on_answer_msg("x3", Mgm2ResponseMessage(True, value=0, gain=10), 1)

        self.assertEqual(computation._state, "gain")
        self.assertEqual(computation._potential_gain, 10)
        self.assertEqual(computation._potential_value, 0)

    def test_response_reject(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._state = "answer?"
        computation._is_offerer = True
        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation.__cost__ = 9
        computation._potential_gain = 9  # best unilateral move
        computation._potential_value = 2  # best unilateral move
        computation._partner = x3

        computation.on_answer_msg("x3", Mgm2ResponseMessage(False), 1)

        self.assertEqual(computation._state, "gain")
        self.assertEqual(computation._potential_gain, 9)
        self.assertEqual(computation._potential_value, 2)
        self.assertEqual(computation._partner, x3)
        self.assertFalse(computation._committed)

    def test_go_accept_no_postponed_value_message(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()
        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation.__cost__ = 9
        computation._state = "go?"
        # from Response message or accepted offer
        computation._potential_gain = 10
        computation._potential_value = 0
        # Common behavior: clear agent view
        computation.on_go_msg("x3", Mgm2GoMessage(True), 1)
        self.assertEqual(computation._state, "value")
        self.test_clear_agent()

        # If cannot move
        self.assertEqual(computation.current_value, 1)
        # If can move
        computation._state = "go?"
        computation._can_move = True
        computation._potential_value = 0
        computation.on_go_msg("x3", Mgm2GoMessage(True), 1)
        self.assertEqual(computation.current_value, 0)

    def test_go_accept_with_postponed_value_message(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation.__cost__ = 9
        computation._state = "go?"
        computation._postponed_msg["value"] = [("x2", Mgm2ValueMessage(1), 1)]
        # from Response message or accepted offer
        computation._potential_gain = 10
        computation._potential_value = 0

        computation.on_go_msg("x3", Mgm2GoMessage(True), 1)
        # Common tests
        self.assertEqual(computation._state, "value")
        self.assertEqual(computation._potential_gain, 0)
        self.assertIsNone(computation._potential_value)
        self.assertEqual(computation._neighbors_values, {"x2": 1})
        self.assertEqual(computation._neighbors_gains, dict())
        self.assertEqual(computation._offers, [])
        self.assertIsNone(computation._partner)
        self.assertEqual(computation.__nb_received_offers__, 0)
        self.assertFalse(computation._committed)
        self.assertFalse(computation._is_offerer)
        self.assertFalse(computation._can_move)

        # If cannot move
        self.assertEqual(computation.current_value, 1)
        # If can move
        computation._can_move = True
        computation._state = "go?"
        computation._potential_value = 0
        computation.on_go_msg("x3", Mgm2GoMessage(True), 1)
        self.assertEqual(computation.current_value, 0)

    def test_go_reject_no_postponed_value_message(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation._potential_value = 0
        computation._state = "go?"
        # from Response message or accepted offer

        computation._handle_go_message("x3", Mgm2GoMessage(False))

        self.assertEqual(computation._state, "value")
        self.assertEqual(computation.__value__, 1)
        self.test_clear_agent()

    def test_go_reject_with_postponed_value_message(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation._state = "go?"
        computation._postponed_msg["value"] = [("x2", Mgm2ValueMessage(1), 1)]
        # from Response message or accepted offer
        computation._potential_value = 0

        computation.on_go_msg("x3", Mgm2GoMessage(False), 1)

        self.assertEqual(computation._state, "value")
        self.assertEqual(computation._state, "value")
        self.assertEqual(computation._potential_gain, 0)
        self.assertIsNone(computation._potential_value)
        self.assertEqual(computation._neighbors_values, {"x2": 1})
        self.assertEqual(computation._neighbors_gains, dict())
        self.assertEqual(computation._offers, [])
        self.assertIsNone(computation._partner)
        self.assertEqual(computation.__nb_received_offers__, 0)
        self.assertFalse(computation._committed)
        self.assertFalse(computation._is_offerer)
        self.assertFalse(computation._can_move)
        self.assertEqual(computation.__value__, 1)

    def test_gain_not_all_received(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._state = "gain"
        computation.on_gain_msg("x2", Mgm2GainMessage(5), 1)

        self.assertEqual(computation._neighbors_gains, {"x2": 5})

    def test_gain_all_received(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._neighbors_values = {"x2": 1, "x3": 0}

        # If potential gain is 0
        computation.__value__ = 1
        computation.__cost__ = 1
        computation._potential_value = 0
        computation._potential_gain = 0
        computation._state = "gain"
        computation._neighbors_gains["x3"] = 2
        computation.on_gain_msg("x2", Mgm2GainMessage(5), 1)
        self.assertEqual(computation.current_value, 1)
        self.assertEqual(computation._state, "value")
        self.assertEqual(computation.current_cost, 1)
        # If commited and has best gain
        computation._state = "gain"
        computation.__value__ = 1
        computation.__cost__ = 1
        computation._committed = True
        computation._partner = x3
        computation._potential_gain = 10
        computation._neighbors_gains["x3"] = 10
        computation._potential_value = 0
        computation.on_gain_msg("x2", Mgm2GainMessage(5), 1)
        self.assertEqual(computation.current_value, 1)
        self.assertEqual(computation.current_cost, 1)
        self.assertTrue(computation._can_move)
        self.assertEqual(computation._state, "go?")
        # If commited and has not best gain
        computation._state = "gain"
        computation.__value__ = 1
        computation.__cost__ = 1
        computation._committed = True
        computation._partner = x3
        computation._potential_gain = 1
        computation._neighbors_gains["x3"] = 1
        computation._potential_value = 0
        computation.on_gain_msg("x2", Mgm2GainMessage(5), 1)
        self.assertEqual(computation.current_value, 1)
        self.assertEqual(computation.current_cost, 1)
        self.assertFalse(computation._can_move)
        self.assertEqual(computation._state, "go?")
        self.test_clear_agent()

        # If not committed and has best gain not alone: no test as it could (in
        # the future) be randomly chosen

        # If not committed and not best gain
        computation._state = "gain"
        computation.__value__ = 1
        computation.__cost__ = 1
        computation._committed = False
        computation._partner = None
        computation._potential_gain = 2
        computation._neighbors_gains["x3"] = 2
        computation._potential_value = 0
        computation.on_gain_msg("x2", Mgm2GainMessage(5), 1)
        self.assertEqual(computation.current_value, 1)
        self.assertEqual(computation.current_cost, 1)
        self.assertEqual(computation._state, "value")
        self.test_clear_agent()

    def test_clear_agent(self):
        x1 = Variable("x1", list(range(3)))
        x2 = Variable("x2", list(range(2)))
        x3 = Variable("x3", list(range(2)))

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        @AsNAryFunctionRelation(x1, x3)
        def psi(x1_, x3_):
            if x1_ == x3_:
                return 8
            return 0

        computation = Mgm2Computation(
            ComputationDef(
                VariableComputationNode(x1, [phi, psi]),
                AlgorithmDef.build_with_default_param("mgm2"),
            )
        )
        computation.message_sender = DummySender()

        computation._neighbors_values = {"x2": 1, "x3": 1}
        computation.__value__ = 1
        computation.__nb_received_offers__ = 2
        computation._partner = x3
        computation._state = "go?"
        computation._potential_gain = 10
        computation._potential_value = 10
        computation._neighbors_values = {"x2": 1, "x3": 0}
        computation._neighbors_gains = {"x2": 5, "x3": 1}
        computation._offers = [(1, 1, "x2")]
        computation._committed = True
        computation._is_offerer = True
        computation._can_move = True

        computation._clear_agent()

        self.assertEqual(computation._potential_gain, 0)
        self.assertEqual(computation._neighbors_values, dict())
        self.assertEqual(computation._neighbors_gains, dict())
        self.assertEqual(computation._offers, [])
        self.assertIsNone(computation._partner)
        self.assertEqual(computation.__nb_received_offers__, 0)
        self.assertFalse(computation._committed)
        self.assertFalse(computation._is_offerer)
        self.assertFalse(computation._can_move)
        self.assertIsNone(computation._potential_value)
        self.assertIsNotNone(computation.current_value)
