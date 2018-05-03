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

from pydcop.computations_graph.objects import Link, ComputationGraph, \
    ComputationNode
from pydcop.infrastructure.discovery import Discovery
from pydcop.reparation.removal import _removal_orphaned_computations, \
    _removal_candidate_agents, _removal_candidate_computations_for_agt, \
    _removal_candidate_computation_info, _removal_candidate_agt_info


class ReparationMessagesTests(unittest.TestCase):

    def setUp(self):

        # A grid-shaped  (3x2) computation graph with 6 computations
        self.l1 = Link(['c1', 'c2'])
        self.l2 = Link(['c2', 'c3'])
        self.l3 = Link(['c1', 'c4'])
        self.l4 = Link(['c2', 'c5'])
        self.l5 = Link(['c3', 'c6'])
        self.l6 = Link(['c4', 'c5'])
        self.l7 = Link(['c5', 'c6'])
        self.links = [self.l1, self.l2, self.l3, self.l4,
                      self.l5, self.l6, self.l7]

        nodes = {}
        for i in range(1, 7):
            name = 'c'+str(i)
            nodes[name] = ComputationNode(
                name, 'test',
                links=[l for l in self.links if l.has_node(name)])

        self.cg = ComputationGraph('test',
                                   nodes=nodes.values())
        # setattr(self.cg, 'links', [self.l1, self.l2, self.l3, self.l4,
        #                            self.l5, self.l6, self.l7])
        #
        # 6 agents hosting these computations
        d = Discovery('a1', 'addr1')
        d.register_computation('c1', 'a1', 'addr1', publish=False)
        d.register_computation('c2', 'a2', 'addr2', publish=False)
        d.register_computation('c3', 'a3', 'addr3', publish=False)
        d.register_computation('c4', 'a4', 'addr4', publish=False)
        d.register_computation('c5', 'a5', 'addr5', publish=False)
        d.register_computation('c6', 'a8', 'addr8', publish=False)
        # and the corresponding replica, 2 for each computation
        d.register_replica('c1', 'a2')
        d.register_replica('c1', 'a5')
        d.register_replica('c2', 'a3')
        d.register_replica('c2', 'a6')
        d.register_replica('c3', 'a1')
        d.register_replica('c3', 'a4')
        d.register_replica('c4', 'a2')
        d.register_replica('c4', 'a5')
        d.register_replica('c5', 'a3')
        d.register_replica('c5', 'a6')
        d.register_replica('c6', 'a1')
        d.register_replica('c6', 'a4')
        self.discovery = d

    def test_orphaned_computation(self):
        # a1 is removed
        orphans = _removal_orphaned_computations(['a1'], self.discovery)
        self.assertEqual(orphans, ['c1'])

        # both a1 and a2 are removed
        orphans = _removal_orphaned_computations(['a1', 'a2'], self.discovery)
        self.assertSetEqual(set(orphans), {'c1', 'c2'})

    def test_removal_candidate_agents(self):
        # a1 is removed
        candidate_computations = _removal_candidate_agents(
            ['a1'], self.discovery)

        self.assertSetEqual(set(candidate_computations),
                            {'a2', 'a5'})

        # both a1 and a2 are removed
        candidate_computations = _removal_candidate_agents(
            ['a1', 'a2'], self.discovery)

        self.assertSetEqual(set(candidate_computations),
                            {'a3', 'a6', 'a5'})

    def test_removal_candidate_computations_for_agt(self):
        # a1 is removed
        orphaned = _removal_orphaned_computations(['a1'], self.discovery)
        candidate_comps = _removal_candidate_computations_for_agt(
            'a2', orphaned, self.discovery)
        self.assertSetEqual(set(candidate_comps), {'c1'})

        # a1 and a4 are removed
        orphaned = _removal_orphaned_computations(['a1', 'a4'], self.discovery)
        candidate_comps = _removal_candidate_computations_for_agt(
            'a2', orphaned, self.discovery)
        self.assertSetEqual(set(candidate_comps), {'c1', 'c4'})
        candidate_comps = _removal_candidate_computations_for_agt(
            'a5', orphaned, self.discovery)
        self.assertSetEqual(set(candidate_comps), {'c1', 'c4'})

    def test_removal_candidate_computation_info(self):
        # remove a1
        agts, fixed, cand = _removal_candidate_computation_info(
            'c1', ['a1'], self.cg, self.discovery)
        self.assertSetEqual(set(agts), {'a2', 'a5'})
        self.assertEqual(fixed['c2'], 'a2')
        self.assertEqual(fixed['c4'], 'a4')
        self.assertFalse(cand)

        # remove a1 and a2
        agts, fixed, cand = _removal_candidate_computation_info(
            'c1', ['a1', 'a2'], self.cg, self.discovery)
        self.assertSetEqual(set(agts), {'a5'})
        self.assertEqual(fixed['c4'], 'a4')
        self.assertNotIn('c2', fixed)
        self.assertSetEqual(set(cand['c2']), {'a3', 'a6'})

    def test_removal_candidate_agt_info(self):

        # remove a1, check info for a2
        infos = _removal_candidate_agt_info('a2', ['a1'],
                                            self.cg, self.discovery)
        self.assertIn('c1', infos)
        agts, fixed, cand = infos['c1']
        self.assertSetEqual(set(agts), {'a2', 'a5'})
        self.assertEqual(fixed['c2'], 'a2')
        self.assertEqual(fixed['c4'], 'a4')
        self.assertFalse(cand)

        # remove a1 and a2, check info for a6
        infos = _removal_candidate_agt_info('a6', ['a1', 'a2'],
                                            self.cg, self.discovery)
        self.assertIn('c2', infos)
        agts, fixed, cand = infos['c2']
        self.assertSetEqual(set(agts), {'a3', 'a6'})
        self.assertEqual(fixed['c3'], 'a3')
        self.assertEqual(fixed['c5'], 'a5')
        self.assertSetEqual(set(cand['c1']), {'a5'})
