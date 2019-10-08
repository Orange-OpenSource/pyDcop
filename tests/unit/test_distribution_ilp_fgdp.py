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

from pydcop.algorithms import amaxsum as ms
from pydcop.algorithms.amaxsum import communication_load, computation_memory
from pydcop.algorithms.maxsum import VARIABLE_UNIT_SIZE
from pydcop.computations_graph.factor_graph import ComputationsFactorGraph, \
    VariableComputationNode, FactorComputationNode, FactorGraphLink
from pydcop.dcop.objects import Variable, VariableDomain, AgentDef
from pydcop.dcop.relations import relation_from_str
from pydcop.distribution.ilp_fgdp import distribute, _build_alphaijk_binvars, \
    _objective_function, _computation_memory_in_cg
from pydcop.distribution.objects import ImpossibleDistributionException

Agent = namedtuple('Agent', ['name'])

d1 = VariableDomain('d1', '', [1, 2, 3, 5])
v1 = Variable('v1', d1)
v2 = Variable('v2', d1)
v3 = Variable('v3', d1)
v4 = Variable('v4', d1)
v5 = Variable('v5', d1)


a1 = AgentDef('a1', capacity=200)
a2 = AgentDef('a2', capacity=200)


def is_all_hosted(cg, dist):
    if len(cg.nodes) == len(dist.computations):
        for c in cg.nodes:
            if c.name not in dist.computations:
                return False
        return True
    return False


class TestDistributionLPFactorGraphWithHints(unittest.TestCase):

    def setUp(self):
        global d1, v1, v2, v3, v4, v5, a1, a2
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        v4 = Variable('v4', d1)
        v5 = Variable('v5', d1)

        a1 = AgentDef('a1', capacity=200)
        a2 = AgentDef('a2', capacity=200)

    def test_raises_if_methods_not_given(self):
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        self.assertRaises(ImpossibleDistributionException, distribute, cg,
                          [Agent('a1'), Agent('a2')], hints=None)

    def test_respect_must_host_for_var(self):

        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"v1" : 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        self.assertEqual(agent_mapping.agent_for('v1'), 'a1')

    def test_respect_must_host_for_fac(self):

        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"f1" : 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')

    def test_respect_must_host_for_fac_and_var(self):

        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"f1" : 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"v1": 0})

        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('v1'), 'a2')

    def test_respect_must_host_for_fac_and_var_same_agent(self):

        f1 = relation_from_str('f1', 'v1 * 0.5 + v2', [v1, v2])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, [])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"f1" : 0, "v1": 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('v1'), 'a1')

    def test_respect_must_host_all_computation_fixed(self):

        f1 = relation_from_str('f1', 'v1 * 0.5 + v2', [v1, v2])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, [])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"f1" : 0, "v1": 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"v2": 0})

        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('v1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('v2'), 'a2')

    def test_respect_must_host_all_computation_invalid(self):

        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"f1" : 0, "v1": 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        # These hints lead to an impossible distribution, as ilp-fgdp requires
        # each agent to host at least one computation. Here Both
        # computations are hosted on a1 and there is no computation
        # available for a2 !
        self.assertRaises(ImpossibleDistributionException, distribute,
                          cg, [a1, a2], hints=None,
                          computation_memory=ms.computation_memory,
                          communication_load=ms.communication_load)

class ILPFGDP(unittest.TestCase):

    def setUp(self):
        global d1, v1, v2, v3, v4, v5, a1, a2
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        v4 = Variable('v4', d1)
        v5 = Variable('v5', d1)

        a1 = AgentDef('a1', capacity=200)
        a2 = AgentDef('a2', capacity=200)

    def test_comm(self):
        f1 = relation_from_str('f1', 'v1 * 0.5 + v2 + v3', [v1, v2, v3])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, ['f1'])
        cv3 = VariableComputationNode(v3, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2, cv3], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"v1" : 0, "v2": 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        a1.capacity = 1000
        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        # As there is enough capacity on a1, factor f1 must go there (where
        # most of its variable are already hosted) while v3 must go on a2 to
        # make sure that all agents are used
        self.assertEqual(agent_mapping.agent_for('f1'), 'a1')
        self.assertEqual(agent_mapping.agent_for('v3'), 'a2')

    def test_comm_not_enough_place(self):
        f1 = relation_from_str('f1', 'v1 * 0.5 + v2 + v3', [v1, v2, v3])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, ['f1'])
        cv3 = VariableComputationNode(v3, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2, cv3], [cf1])

        a1 = AgentDef("a1", capacity=200,  default_hosting_cost=1,
                 hosting_costs={"v1" : 0, "v2": 0})

        a2 = AgentDef("a2", capacity=200,  default_hosting_cost=1)

        a1.capacity = 15
        agent_mapping = distribute(cg, [a1, a2],
                                   hints=None,
                                   computation_memory=ms.computation_memory,
                                   communication_load=ms.communication_load)

        # As there is enough not capacity on a1, factor f1 and variable v3
        # must go on a2 
        self.assertEqual(agent_mapping.agent_for('f1'), 'a2')
        self.assertEqual(agent_mapping.agent_for('v3'), 'a2')


class UtilityFunctions(unittest.TestCase):

    def setUp(self):
        global d1, v1, v2, v3, v4, v5, a1, a2
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        v4 = Variable('v4', d1)
        v5 = Variable('v5', d1)

        a1 = AgentDef('a1', capacity=200)
        a2 = AgentDef('a2', capacity=200)

    def test_build_alphaijk_one_var_one_fac(self):
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        agents_names = ['a1', 'a2']

        alphas = _build_alphaijk_binvars(cg, agents_names)

        self.assertEqual(len(alphas), 2)
        self.assertIn((('v1', 'f1'), 'a1'), alphas)
        self.assertIn((('v1', 'f1'), 'a2'), alphas)
        print(alphas)

    def test_build_alphaijk_one_var_two_fac(self):
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        f2 = relation_from_str('f2', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1', 'f2'])
        cf1 = FactorComputationNode(f1)
        cf2 = FactorComputationNode(f2)
        cg = ComputationsFactorGraph([cv1], [cf1, cf2])

        agents_names = ['a1', 'a2']

        alphas = _build_alphaijk_binvars(cg, agents_names)

        self.assertEqual(len(alphas), 4)
        self.assertIn((('v1', 'f1'), 'a1'), alphas)
        self.assertIn((('v1', 'f2'), 'a1'), alphas)
        self.assertIn((('v1', 'f1'), 'a2'), alphas)
        self.assertIn((('v1', 'f2'), 'a2'), alphas)
        print(alphas)

    def test_obj_function(self):
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        agents_names = ['a1', 'a2']
        alphas = _build_alphaijk_binvars(cg, agents_names)

        obj = _objective_function(cg, communication_load, alphas, agents_names)

        # In that case, the objective function must depend on two variables:
        self.assertEqual(len(obj.sorted_keys()), 2)
        # print(obj)
        # print(list(obj.sorted_keys()))


class ComputationMemory(unittest.TestCase):

    def setUp(self):
        global d1, v1, v2, v3, v4, v5, a1, a2
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        v3 = Variable('v3', d1)
        v4 = Variable('v4', d1)
        v5 = Variable('v5', d1)

        a1 = AgentDef('a1', capacity=200)
        a2 = AgentDef('a2', capacity=200)

    def test_var_fac_nolink(self):
        f1 = relation_from_str('f1', '0.5', [])
        cv1 = VariableComputationNode(v1, [])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        self.assertEqual(_computation_memory_in_cg('v1', cg,
                                                   computation_memory), 0)
        self.assertEqual(_computation_memory_in_cg('f1',
                                                   cg, computation_memory),
                         0)

    def test_var_fac_link(self):
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        # size of the domain of v1
        self.assertEqual(_computation_memory_in_cg('v1', cg,
                                                   computation_memory),
                         4 *  VARIABLE_UNIT_SIZE)
        self.assertEqual(_computation_memory_in_cg('f1', cg,
                                                   computation_memory),
                         4)

    def test_fac_2var(self):
        f1 = relation_from_str('f1', 'v1 * 0.5+v2', [v1, v2])
        cv1 = VariableComputationNode(v1, ['f1'])
        cv2 = VariableComputationNode(v2, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1, cv2], [cf1])

        # size of the domain of v1 + size domain v2
        self.assertEqual(_computation_memory_in_cg('f1', cg,
                                                   computation_memory),
                         8)
