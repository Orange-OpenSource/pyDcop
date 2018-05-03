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

from pydcop.algorithms import ComputationDef, dsa
from pydcop.algorithms.dsa import DsaComputation, build_computation
from pydcop.algorithms.objects import AlgoDef
from pydcop.computations_graph.constraints_hypergraph \
    import VariableComputationNode, ConstraintLink
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import UnaryFunctionRelation, \
    AsNAryFunctionRelation, relation_from_str, constraint_from_str


def test_communication_load():
    v = Variable('v1', list(range(10)))
    var_node = VariableComputationNode(v, [])
    assert dsa.UNIT_SIZE + dsa.HEADER_SIZE == dsa.communication_load(
        var_node, 'f1')


def test_computation_memory_one_constraint():
    v1 = Variable('v1', list(range(10)))
    v2 = Variable('v2', list(range(10)))
    v3 = Variable('v3', list(range(10)))
    c1 = constraint_from_str('c1', ' v1 + v2 == v3', [v1, v2, v3])
    v1_node = VariableComputationNode(v1, [c1])

    # here, we have an hyper-edges with 3 vertices
    assert dsa.computation_memory(v1_node) == dsa.UNIT_SIZE * 2


def test_computation_memory_two_constraints():
    v1 = Variable('v1', list(range(10)))
    v2 = Variable('v2', list(range(10)))
    v3 = Variable('v3', list(range(10)))
    v4 = Variable('v4', list(range(10)))
    c1 = constraint_from_str('c1', ' v1 == v2', [v1, v2])
    c2 = constraint_from_str('c2', ' v1 == v3', [v1, v3])
    c3 = constraint_from_str('c3', ' v1 == v4', [v1, v4])
    v1_node = VariableComputationNode(v1, [c1, c2, c3])

    # here, we have 3 edges , one for each constraint
    assert dsa.computation_memory(v1_node) == dsa.UNIT_SIZE * 3


def test_footprint_on_computation_object():
    v1 = Variable('v1', [0, 1, 2, 3, 4])
    v2 = Variable('v2', [0, 1, 2, 3, 4])
    c1 = relation_from_str('c1', '0 if v1 == v2 else  1', [v1, v2])
    n1 = VariableComputationNode(v1, [c1])
    n2 = VariableComputationNode(v2, [c1])
    comp_def = ComputationDef(n1, AlgoDef('dsa', mode='min'))
    c = build_computation(comp_def)

    assert c.footprint() ==  5


class Neighbors(unittest.TestCase):

    def test_1_unary_constraint_means_no_neighbors(self):
        variable = Variable('a', [0, 1, 2, 3, 4])
        c1 = UnaryFunctionRelation('c1', variable, lambda x: abs(x - 2))

        computation = DsaComputation(variable, [c1], comp_def=MagicMock())
        self.assertEqual(len(computation._neighbors), 0)

    def test_2_unary_constraint_means_no_neighbors(self):
        variable = Variable('a', [0, 1, 2, 3, 4])
        c1 = UnaryFunctionRelation('c1', variable, lambda x: abs(x - 3))
        c2 = UnaryFunctionRelation('c1', variable, lambda x: abs(x - 1) * 2)

        computation = DsaComputation(variable, [c1, c2], comp_def=MagicMock())
        self.assertEqual(len(computation._neighbors), 0)

    def test_one_binary_constraint_one_neighbors(self):

        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2)
        def c1(v1_, v2_):
            return abs(v1_ - v2_)

        computation = DsaComputation(v1, [c1], comp_def=MagicMock())
        self.assertEqual(len(computation._neighbors), 1)

    def test_2_binary_constraint_one_neighbors(self):
        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2)
        def c1(v1_, v2_):
            return abs(v1_ - v2_)

        @AsNAryFunctionRelation(v1, v2)
        def c2(v1_, v2_):
            return abs(v1_ + v2_)

        computation = DsaComputation(v1, [c1, c2], comp_def=MagicMock())
        self.assertEqual(len(computation._neighbors), 1)

    def test_3ary_constraint_2_neighbors(self):
        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])
        v3 = Variable('v3', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2, v3)
        def c1(v1_, v2_, v3_):
            return abs(v1_ - v2_ + v3_)

        computation = DsaComputation(v1, [c1], comp_def=MagicMock())
        self.assertEqual(len(computation._neighbors), 2)


class CalcBestVal(unittest.TestCase):

    def test_1_unary_constraint(self):

        variable = Variable('a', [0, 1, 2, 3, 4])
        c1 = UnaryFunctionRelation('c1', variable, lambda x: abs(x-2))

        computation = DsaComputation(variable, [c1], comp_def=MagicMock())
        val, sum_costs = computation._compute_best_value()

        self.assertEqual(val, [2])
        self.assertEqual(sum_costs, 0)

    def test_2_unary_constraints(self):
        variable = Variable('a', [0, 1, 2, 3, 4])
        c1 = UnaryFunctionRelation('c1', variable, lambda x: abs(x - 3))
        c2 = UnaryFunctionRelation('c1', variable, lambda x: abs(x-1)*2)

        computation = DsaComputation(variable, [c1, c2], comp_def=MagicMock())
        val, sum_costs = computation._compute_best_value()
        print(val, sum_costs)

        self.assertEqual(val, [1])
        self.assertEqual(sum_costs, 2)

    def test_1_binary_constraint(self):

        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2)
        def c1(v1_, v2_):
            return abs(v1_-v2_)

        computation = DsaComputation(v1, [c1], comp_def=MagicMock())
        computation._neighbors_values = {'v1': 1}
        val, sum_costs = computation._compute_best_value()

        self.assertEqual(val, [1])
        self.assertEqual(sum_costs, 0)


class ComputeBoundary(unittest.TestCase):

    def test_mode_is_min(self):
        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])
        v3 = Variable('v3', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2)
        def c1(v1_, v2_):
            return abs(v1_-v2_)

        @AsNAryFunctionRelation(v1, v3)
        def c2(v1_, v3_):
            return abs(v1_-v3_)

        computation = DsaComputation(v1, [c1, c2], comp_def=MagicMock())
        c, b = computation._compute_boundary([c1, c2])

        self.assertEqual(c, [c1, c2])
        self.assertEqual(b, 0)

    def test_mode_is_max(self):
        v1 = Variable('v1', [0, 1, 2, 3, 4])
        v2 = Variable('v2', [0, 1, 2, 3, 4])
        v3 = Variable('v3', [0, 1, 2, 3, 4])

        @AsNAryFunctionRelation(v1, v2)
        def c1(v1_, v2_):
            return abs(v1_-v2_)

        @AsNAryFunctionRelation(v1, v3)
        def c2(v1_, v3_):
            return abs(v1_-v3_)

        computation = DsaComputation(v1, [c1, c2], mode="max",
                                     comp_def=MagicMock())
        c, b = computation._compute_boundary([c1, c2])

        self.assertEqual(c, [c1, c2])
        self.assertEqual(b, 8)
