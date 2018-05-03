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
    VariableComputationNode, FactorComputationNode, FactorGraphLink
from pydcop.dcop.objects import Variable, VariableDomain, AgentDef
from pydcop.dcop.relations import relation_from_str
from pydcop.distribution.objects import ImpossibleDistributionException
from pydcop.distribution.oneagent import distribute


Agent = namedtuple('Agent', ['name'])

class TestDistributionOneAgent(unittest.TestCase):

    def test_simple_fg(self):
        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        agents= [AgentDef('a1'), AgentDef('a2')]
        distribution = distribute(cg, agents)

        self.assertEqual(len(distribution.agents), 2)
        self.assertEqual(len(distribution.computations), 2)
        self.assertEqual(len(distribution.computations_hosted('a1')), 1)
        self.assertEqual(len(distribution.computations_hosted('a2')), 1)
        self.assertNotEqual(distribution.computations_hosted('a2'),
                            distribution.computations_hosted('a1'))

    def test_raise_when_not_enough_agents(self):

        d1 = VariableDomain('d1', '', [1, 2, 3, 5])

        v1 = Variable('v1', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])
        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

        self.assertRaises(ImpossibleDistributionException, distribute, cg,
                          [AgentDef('a1')])
