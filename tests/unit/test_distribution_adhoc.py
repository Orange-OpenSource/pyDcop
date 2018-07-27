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
from collections import namedtuple

from pydcop.computations_graph.factor_graph import ComputationsFactorGraph, \
    VariableComputationNode, FactorComputationNode
from pydcop.dcop.objects import Variable, VariableDomain, AgentDef
from pydcop.dcop.relations import relation_from_str
from pydcop.distribution.adhoc import distribute
from pydcop.distribution.objects import DistributionHints


Agent = namedtuple('Agent', ['name'])


class TestDistributionAdHocFactorGraph(unittest.TestCase):

    def test_no_hints(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])
        agents = [AgentDef('a1', capacity=100), AgentDef('a2', capacity=100)]
        agent_mapping = distribute(cg, agents, hints=None,
                                   computation_memory=lambda x: 10)
        self.assertTrue(agent_mapping.is_hosted(['v1', 'f1']))

    def test_must_host_one(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        hints = DistributionHints({'a1': ['v1']}, None)
        agents = [AgentDef('a1', capacity=100), AgentDef('a2', capacity=100)]
        agent_mapping = distribute(cg, agents, hints,
                                   computation_memory=lambda x: 10)

        self.assertIn('v1', agent_mapping.computations_hosted('a1'))
        self.assertTrue(is_all_hosted(cg, agent_mapping))

    def test_must_host_two_vars(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, [])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        hints = DistributionHints({'a1': ['v1', 'v2']}, None)
        agents = [AgentDef('a1', capacity=100), AgentDef('a2', capacity=100)]
        agent_mapping = distribute(cg, agents, hints,
                                   computation_memory=lambda x: 10)

        self.assertIn('v1', agent_mapping.computations_hosted('a1'))
        self.assertIn('v2', agent_mapping.computations_hosted('a1'))
        self.assertTrue(is_all_hosted(cg, agent_mapping))

    def test_must_host_two_agents(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, [])
        cf1 = FactorComputationNode(f1)
        # link = FactorGraphLink('f1', 'v1')
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        hints = DistributionHints({'a1': ['v1'], 'a2': ['v2']}, None)
        agents = [AgentDef('a1', capacity=100), AgentDef('a2', capacity=100)]
        agent_mapping = distribute(cg, agents, hints,
                                   computation_memory=lambda x: 10)

        self.assertIn('v1', agent_mapping.computations_hosted('a1'))
        self.assertIn('v2', agent_mapping.computations_hosted('a2'))
        self.assertTrue(is_all_hosted(cg, agent_mapping))

    def test_host_with(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, [])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        hints = DistributionHints(None, {'v1': ['f1']})

        agents = [AgentDef('a{}'.format(i), capacity=100)
                  for i in range(1, 11)]
        agent_mapping = distribute(cg, agents, hints,
                                   computation_memory=lambda x: 10)

        self.assertEqual(agent_mapping.agent_for('v1'),
                         agent_mapping.agent_for('f1'))
        self.assertTrue(is_all_hosted(cg, agent_mapping))

    def test_host_on_highest_dependent_agent(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        f1 = relation_from_str('f1', 'v1 + v2', [v1, v2])
        f2 = relation_from_str('f2', 'v1 - v2 + v3', [v1, v2, v3])
        cv1 = VariableComputationNode(v1, ['f1', 'f2'])
        cv2 = VariableComputationNode(v2, ['f1', 'f2'])
        cv3 = VariableComputationNode(v3, ['f2'])
        cf1 = FactorComputationNode(f1)
        cf2 = FactorComputationNode(f2)
        cg = ComputationsFactorGraph([cv1, cv2, cv3], [cf1, cf2])

        hints = DistributionHints(must_host={'a1': ['v1'], 'a2': ['v2', 'v3']})

        # we must set the capacity to make sure that a2 cannot take f1
        agents = [AgentDef('a{}'.format(i), capacity=41)
                  for i in range(1, 11)]

        agent_mapping = distribute(cg, agents, hints,
                                   computation_memory=lambda x: 10)

        print(agent_mapping)
        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('f2'), 'a2')

        self.assertTrue(is_all_hosted(cg, agent_mapping))


class TestDistributionAdHocFactorGraphSecp(unittest.TestCase):

    def setUp(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        # An secp made of 2 lights, one model and two rule
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        m1 = Variable('m1', d1)
        mf1 = relation_from_str('mf1', 'v1 + v2 == m1', [v1, v2, m1])
        r1 = relation_from_str('r1', 'm1 - v2', [m1, v2])
        r2 = relation_from_str('r2', 'v3', [v3])
        cv1 = VariableComputationNode(v1, ['mf1'])
        cv2 = VariableComputationNode(v2, ['mf1', 'r1'])
        cv3 = VariableComputationNode(v3, ['r2'])
        cm1 = VariableComputationNode(m1, ['mf1', 'r1'])
        cmf1 = FactorComputationNode(mf1)
        cr1 = FactorComputationNode(r1)
        cr2 = FactorComputationNode(r2)
        self.cg = ComputationsFactorGraph([cv1, cv2, cv3, cm1],
                                          [cmf1, cr1, cr2])

    def test_model_on_single_agent(self):

        hints = DistributionHints(must_host={'a1': ['v1'], 'a2': ['v2']},
                                  host_with={'m1': ['mf1']})

        agents = [AgentDef('a{}'.format(i), capacity=100) for i in range(1, 11)]
        agent_mapping = distribute(self.cg, agents, hints,
                                   computation_memory=lambda x: 10)

        # Check that the variable and relation of the model are on the same
        # agent
        self.assertEqual(agent_mapping.agent_for('m1'),
                         agent_mapping.agent_for('mf1'))

        self.assertTrue(is_all_hosted(self.cg, agent_mapping))

    def test_model_on_dependent_light(self):

        hints = DistributionHints(must_host={'a1': ['v1'], 'a2': ['v2']},
                                  host_with={'m1': ['mf1']})

        agents = [AgentDef('a{}'.format(i), capacity=100) for i in range(1, 11)]
        agent_mapping = distribute(self.cg, agents, hints,
                                   computation_memory=lambda x: 10)

        # Check that the variable and relation of the model are on the same
        # agent
        self.assertIn(agent_mapping.agent_for('m1'), ['a1', 'a2'])
        self.assertIn(agent_mapping.agent_for('mf1'), ['a1', 'a2'])

        self.assertTrue(is_all_hosted(self.cg, agent_mapping))

    def test_rule_with_model(self):

        hints = DistributionHints(must_host={'a1': ['v1'], 'a3': ['v2']},
                                  host_with={'m1': ['mf1']})

        agents = [AgentDef('a{}'.format(i), capacity=100) for i in range(1, 11)]
        agent_mapping = distribute(self.cg, agents, hints,
                                   computation_memory=lambda x: 10)

        # rule should be hosted either with model m1 or variable v2
        # print(agent_mapping.agent_for('m1'), agent_mapping.agent_for('mf1'),
        #       agent_mapping.agent_for('v1'), agent_mapping.agent_for('v2'),
        #       agent_mapping.agent_for('r1'))
        self.assertTrue(agent_mapping.agent_for('r1') ==
                        agent_mapping.agent_for('v2')
                        or
                        agent_mapping.agent_for('m1') ==
                        agent_mapping.agent_for('r1'))

        self.assertTrue(is_all_hosted(self.cg, agent_mapping))

    def test_rule_with_light(self):

        hints = DistributionHints(must_host={'a1': ['v1'], 'a3': ['v2']},
                                  host_with={'m1': ['mf1']})

        agents = [AgentDef('a{}'.format(i), capacity=100) for i in range(1, 11)]
        agent_mapping = distribute(self.cg, agents, hints,
                                   computation_memory=lambda x: 10)

        # rule r2 only depends on v3, it must be hosted on the same agent
        self.assertEqual(agent_mapping.agent_for('v3'),
                         agent_mapping.agent_for('r2'))

        self.assertTrue(is_all_hosted(self.cg, agent_mapping))


def is_all_hosted(cg, dist):
    if len(cg.nodes) == len(dist.computations):
        for c in cg.nodes:
            if c.name not in dist.computations:
                return False
        return True
    return False
