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


import json
import unittest
from unittest.mock import MagicMock

from pydcop.algorithms.maxsum import (
    approx_match,
    FactorAlgo,
    computation_memory,
    VARIABLE_UNIT_SIZE,
    FACTOR_UNIT_SIZE,
    communication_load,
    HEADER_SIZE,
    UNIT_SIZE,
    MaxSumMessage,
)
from pydcop.computations_graph.factor_graph import (
    VariableComputationNode,
    FactorComputationNode,
)
from pydcop.dcop.objects import Variable, VariableDomain
from pydcop.dcop.relations import AsNAryFunctionRelation, relation_from_str
from pydcop.utils.simple_repr import simple_repr, from_repr


class MaxSumFactorAlgoTest(unittest.TestCase):
    def test_init(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        comp_def = MagicMock()
        comp_def.algo.algo = "maxsum"
        comp_def.algo.mode = "min"
        f = FactorAlgo(phi, comp_def=comp_def)

        self.assertEqual(f.name, "phi")
        self.assertEqual(len(f.variables), 2)

    def test_cost_for_1var(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)

        @AsNAryFunctionRelation(x1)
        def cost(x1_):
            return x1_ * 2

        comp_def = MagicMock()
        comp_def.algo.algo = "maxsum"
        comp_def.algo.mode = "min"
        f = FactorAlgo(cost, comp_def=comp_def)

        costs = f._costs_for_var(x1)

        # in the max-sum algorithm, for an unary factor the costs is simply
        # the result of the factor function
        self.assertEqual(costs[0], 0)
        self.assertEqual(costs[5], 10)

    def test_cost_for_1var_2(self):

        # TODO test for min and max

        domain = list(range(10))
        x1 = Variable("x1", domain)

        @AsNAryFunctionRelation(x1)
        def cost(x1):
            return x1 * 2

        comp_def = MagicMock()
        comp_def.algo.algo = "maxsum"
        comp_def.algo.mode = "min"
        f = FactorAlgo(cost, comp_def=comp_def)

        costs = f._costs_for_var(x1)

        # in the maxsum algorithm, for an unary factor the costs is simply
        # the result of the factor function
        self.assertEqual(costs[0], 0)
        self.assertEqual(costs[5], 10)

    def test_cost_for_2var(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        domain = list(range(5))
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def cost(x1_, x2_):
            return abs((x1_ - x2_) / 2)

        comp_def = MagicMock()
        comp_def.algo.algo = "maxsum"
        comp_def.algo.mode = "min"
        f = FactorAlgo(cost, comp_def=comp_def)

        costs = f._costs_for_var(x1)

        # in this test, the factor did not receive any costs messages from
        # other variables, this means it  only uses the factor function when
        # calculating costs.

        # x1 = 5, best val for x2 is 4, with cost = 0.5
        self.assertEqual(costs[5], (5 - 4) / 2)
        self.assertEqual(costs[9], (9 - 4) / 2)
        self.assertEqual(costs[2], 0)


class VarDummy:
    def __init__(self, name):
        self.name = name
        self.current_value = None
        self.current_cost = None


class ApproxMatchTests(unittest.TestCase):
    def test_match_exact(self):
        c1 = {0: 0, 1: 0, 2: 0}
        c2 = {0: 0, 1: 0, 2: 0}

        self.assertTrue(approx_match(c1, c2))

    def test_nomatch(self):
        c1 = {0: 0, 1: 0, 2: 0}
        c2 = {0: 0, 1: 1, 2: 0}

        self.assertFalse(approx_match(c1, c2))

    def test_nomatch2(self):
        c1 = {
            0: -46.0,
            1: -46.5,
            2: -55.5,
            3: -56.0,
            4: -56.5,
            5: -65.5,
            6: -66.0,
            7: -66.5,
            8: -67.0,
            9: -67.5,
        }
        c2 = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
        }

        self.assertFalse(approx_match(c1, c2))


class ComputationMemory(unittest.TestCase):
    def test_variable_memory_no_neighbor(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 5])
        v1 = Variable("v1", d1)

        vn1 = VariableComputationNode(v1, [])

        # If a variable has no neighbors, it does not need to keep any cost
        # and thus requires no memory
        self.assertEqual(computation_memory(vn1), 0)

    def test_variable_memory_one_neighbor(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 5])
        v1 = Variable("v1", d1)
        f1 = relation_from_str("f1", "v1 * 0.5", [v1])

        cv1 = VariableComputationNode(v1, ["f1"])
        cf1 = FactorComputationNode(f1)

        self.assertEqual(computation_memory(cv1), VARIABLE_UNIT_SIZE * 4)

    def test_factor_memory_one_neighbor(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 5])
        v1 = Variable("v1", d1)
        f1 = relation_from_str("f1", "v1 * 0.5", [v1])

        cv1 = VariableComputationNode(v1, ["f1"])
        cf1 = FactorComputationNode(f1)

        self.assertEqual(computation_memory(cf1), FACTOR_UNIT_SIZE * 4)

    def test_factor_memory_two_neighbor(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 4, 5])
        v1 = Variable("v1", d1)
        d2 = VariableDomain("d1", "", [1, 2, 3])
        v2 = Variable("v2", d2)
        f1 = relation_from_str("f1", "v1 * 0.5 + v2", [v1, v2])

        cv1 = VariableComputationNode(v1, ["f1"])
        cv2 = VariableComputationNode(v2, ["f1"])
        cf1 = FactorComputationNode(f1)

        self.assertEqual(computation_memory(cf1), FACTOR_UNIT_SIZE * (5 + 3))

    def test_variable_memory_two_neighbor(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 5])
        v1 = Variable("v1", d1)
        cv1 = VariableComputationNode(v1, ["f1", "f2"])

        self.assertEqual(computation_memory(cv1), VARIABLE_UNIT_SIZE * 4 * 2)


class CommunicationCost(unittest.TestCase):
    def test_variable_one_neighbors(self):
        d1 = VariableDomain("d1", "", [1, 2, 3, 5])
        v1 = Variable("v1", d1)
        f1 = relation_from_str("f1", "v1 * 0.5", [v1])

        cv1 = VariableComputationNode(v1, ["f1"])
        cf1 = FactorComputationNode(f1)

        # If a variable has no neighbors, it does not need to keep any cost
        # and thus requires no memory
        self.assertEqual(
            communication_load(cv1, "f1"), HEADER_SIZE + UNIT_SIZE * len(v1.domain)
        )
        self.assertEqual(
            communication_load(cf1, "v1"), HEADER_SIZE + UNIT_SIZE * len(v1.domain)
        )


class TestsMaxsumMessage(unittest.TestCase):
    def test_serialize_repr(self):
        # Make sure that even after serialization / deserialization,
        # from_repr and simple_repr still produce equal messages.
        # This has been causing problems with maxsum costs dict where key
        # were integers

        msg = MaxSumMessage({1: 10, 2: 20})
        r = simple_repr(msg)
        msg_json = json.dumps(r)

        r2 = json.loads(msg_json)
        msg2 = from_repr(r2)

        self.assertEqual(msg, msg2)
