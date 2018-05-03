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
from unittest.case import TestCase
from unittest.mock import Mock

from pydcop.computations_graph.constraints_hypergraph import \
    ComputationConstraintsHyperGraph, ConstraintLink, VariableComputationNode
from pydcop.computations_graph.objects import ComputationGraph, \
    ComputationNode, Link
from pydcop.dcop.objects import AgentDef, create_variables, Domain
from pydcop.dcop.relations import constraint_from_str

from pydcop.distribution.ilp_compref import lp_model


class TestLpModelHostingCost(unittest.TestCase):

    def setUp(self):

        c1 = ComputationNode('c1', 'dummy_type', neighbors=['c2'])
        c2 = ComputationNode('c2', 'dummy_type', neighbors=['c1'])

        self.cg = ComputationGraph(graph_type='test',
                                   nodes= [c1, c2])
        self.agents = [AgentDef('a1'),
                       AgentDef('a2')]

    def test_one_comp_on_each(self):

        # let's use some hardcoded value for footprint, capacity, etc.

        # Each agent can hold exactly one computation, there is only two
        # possible dist
        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=lambda a1, a2: 10,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=lambda a, c: 1)

        self.assertEqual(len(mapping['a1']), 1)
        self.assertEqual(len(mapping['a2']), 1)

    def test_one_comp_on_each_with_hosting_cost(self):

        # This time we introduce a clear preference c1-a2 and c2 - a1
        def hosting_cost(a, c):
            if c == 'c1' and a == 'a2':
                return 0
            if c == 'c2' and a == 'a1':
                return 0
            return 5

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=lambda a1, a2: 10,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=hosting_cost)

        self.assertIn('c2', mapping['a1'])
        self.assertIn('c1', mapping['a2'])

    def test_one_comp_on_each_with_pref_competition(self):

        # Both computation are attracted to a2, but c1 is more
        def hosting_cost(a, c):
            if c == 'c1' and a == 'a2':
                return 1
            if c == 'c2' and a == 'a2':
                return 5
            return 10

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=lambda a1, a2: 10,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=hosting_cost)

        self.assertIn('c2', mapping['a1'])
        self.assertIn('c1', mapping['a2'])

    def test_one_comp_on_each_with_pref_competition2(self):

        # Both computation are attrative to a2
        # and a2 has enough capacity for both.
        def hosting_cost(a, c):
            if c == 'c1' and a == 'a2':
                return 5
            if c == 'c2' and a == 'a2':
                return 5
            return 10

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 20,
                           route=lambda a1, a2: 10,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=hosting_cost)

        self.assertIn('c2', mapping['a2'])
        self.assertIn('c1', mapping['a2'])


class TestLpModelMsgLoad(unittest.TestCase):

    def setUp(self):

        c1 = ComputationNode('c1', neighbors=['c2'])
        c2 = ComputationNode('c2', neighbors=['c1'])
        self.cg = ComputationGraph(graph_type='test',
                                   nodes=[c1, c2])
        self.agents = [AgentDef('a1'),
                       AgentDef('a2'),
                       AgentDef('a3')]

    def test_one_costly_route(self):

        # The route between a1 and a2 is more costly than the other routes
        # so computations should be hosted on a1 and a3 or a3 and a2
        def route(a1, a2):
            if is_same_route((a1, a2), ('a1', 'a2')):
                return 5
            return 1

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=route,
                           msg_load=lambda c1, c2: 1,
                           hosting_cost=lambda c, a: 0)

        # hosting on a1 AND a2 is invalid as it would cost more:
        invalid_dist = len(mapping['a1']) == 1 and len(mapping['a2']) == 1
        self.assertFalse(invalid_dist)

    def test_two_costly_routes(self):

        # The routes a1-a2 and a3-a1 are more costly than the other routes
        # to avoid costly route, computation must be hosted on a2 and a3
        def route(a1, a2):
            if is_same_route((a1, a2), ('a1', 'a2')):
                return 5
            if is_same_route((a1, a2), ('a3', 'a1')):
                return 5
            return 1

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=route,
                           msg_load=lambda c1, c2: 1,
                           hosting_cost=lambda c, a: 1)

        # two valid distributions
        valid_dist = ('c1' in mapping['a2'] and 'c2'in mapping['a3']) or \
            ('c1' in mapping['a3'] and 'c2' in mapping['a2'])
        self.assertTrue(valid_dist)


class TestLpModelRouteAndPref(unittest.TestCase):

    def setUp(self):

        c1 = ComputationNode('c1', 'dummy_type', neighbors=['c2'])
        c2 = ComputationNode('c2', 'dummy_type', neighbors=['c1'])

        self.cg = ComputationGraph(graph_type='test',
                                   nodes=[c1, c2])
        self.agents = [AgentDef('a1'),
                       AgentDef('a2'),
                       AgentDef('a3')]

    def test_one_costly_route(self):

        # The route between a1 and a2 is more costly than the other routes
        # so computations should be hosted on a1 and a3 or a3 and a2
        def route(a1, a2):
            if is_same_route((a1, a2), ('a1', 'a2')):
                return 10
            return 1

        # c1 is attracted to a2
        def hosting_cost(a, c):
            if c == 'c1' and a == 'a2':
                return 1
            return 10

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=route,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=hosting_cost)

        # c1 should be hosted on a2 beacause of pref
        # and c2 on a3, to avoid the costly a1-a2 route
        self.assertIn('c1', mapping['a2'])
        self.assertIn('c2', mapping['a3'])


class TestLpModelWithHyperGraph(unittest.TestCase):

    def setUp(self):

        variables = list(create_variables(
            'v', ['1','2','3'], Domain('d', '', [ 1, 2])).values())
        all_diff = constraint_from_str('all_diff', 'v1 + v2 + v3 ', variables)
        v1, v2, v3 = variables
        c1 = VariableComputationNode(v1, [all_diff])
        c2 = VariableComputationNode(v2, [all_diff])
        c3 = VariableComputationNode(v3, [all_diff])
        nodes = [c1, c2, c3]
        # links = [ConstraintLink('all_diff', ['c1', 'c2', 'c3'])]

        self.cg = ComputationConstraintsHyperGraph(nodes)
        self.agents = [AgentDef('a1'),
                       AgentDef('a2'),
                       AgentDef('a3')]

    def test_sharing_computation(self):

        # given their capcity, each agent must host exactly one of the 3
        # computations
        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 10,
                           route=lambda a1, a2: 1,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=lambda c, a: 0)

        self.assertEqual(len(mapping['a1']), 1)
        self.assertEqual(len(mapping['a2']), 1)
        self.assertEqual(len(mapping['a3']), 1)

    def test_group_computations(self):

        # No capcity problem : putting all computations on the same agent is
        # the cheapest option.
        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 100,
                           route=lambda a1, a2: 1,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=lambda c, a: 0)

        self.assertTrue(len(mapping['a1']) == 3 or
                        len(mapping['a2']) == 3 or
                        len(mapping['a3']) == 3)

    def test_split_in_two_groups(self):

        def route(a1, a2):
            if is_same_route((a1, a2), ('a1', 'a2')):
                return 10
            return 1

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 20,
                           route=route,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=lambda c, a: 0)

        # route a1-a2 is costly, either a1 or a2 must be empty
        self.assertTrue(len(mapping['a1']) == 0 or
                        len(mapping['a2']) == 0)

    def test_split_in_two_groups_prefa1(self):

        def route(a1, a2):
            if is_same_route((a1, a2), ('a1', 'a2')):
                return 10
            return 1

        def hosting_costa1(a, _):
            if a == 'a1':
                return 0
            return 10

        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 20,
                           route=route,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=hosting_costa1)

        # Route a1-a2 is costly, either a1 or a2 must be empty
        # as a1 is more attractive, a2 should be empty:
        self.assertTrue(len(mapping['a2']) == 0)

        # And if we make a2 more attractive, a1 is empty:
        mapping = lp_model(self.cg, self.agents,
                           footprint=lambda c: 10,
                           capacity=lambda a: 20,
                           route=route,
                           msg_load=lambda c1, c2: 10,
                           hosting_cost=lambda a, _: 0 if a == 'a2' else 10)

        self.assertTrue(len(mapping['a1']) == 0)


def is_same_route(r1, r2):
    a1, a2 = r2
    return (r1 == r2) or (r1 == (a2, a1))
