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

from pydcop.distribution.objects import Distribution, DistributionHints


class TestDistributionObject(unittest.TestCase):

    def test_dist(self):

        d = Distribution({'a1': ['v1'], 'a2': ['v2']})

        self.assertEqual(len(d.computations_hosted('a1')), 1)
        self.assertEqual(len(d.computations_hosted('a2')), 1)
        self.assertIn('v1', d.computations_hosted('a1'))
        self.assertIn('v2', d.computations_hosted('a2'))

        self.assertEqual(d.agent_for('v1'), 'a1')
        self.assertEqual(d.agent_for('v2'), 'a2')

    def test_dist_2(self):

        d = Distribution({'a1': ['v1', 'v2'], 'a2': ['v3']})

        self.assertEqual(len(d.computations_hosted('a1')), 2)
        self.assertEqual(len(d.computations_hosted('a2')), 1)
        self.assertIn('v1', d.computations_hosted('a1'))
        self.assertIn('v2', d.computations_hosted('a1'))

        self.assertEqual(d.agent_for('v1'), 'a1')
        self.assertEqual(d.agent_for('v2'), 'a1')
        self.assertEqual(d.agent_for('v3'), 'a2')

    def test_raise_on_invalid_mapping(self):

        self.assertRaises(ValueError, Distribution, {'a1': 'v1', 'a2': 'v2'})

    def test_host_on_agent(self):
        d = Distribution({'a1': ['v1', 'v2'], 'a2': ['v3']})
        d.host_on_agent('a1', ['v4'])


        self.assertEqual(d.agent_for('v4'), 'a1')
        self.assertEqual(d.agent_for('v1'), 'a1')

        self.assertIn('v4', d.computations_hosted('a1'))
        self.assertIn('v1', d.computations_hosted('a1'))
        self.assertIn('v2', d.computations_hosted('a1'))

    def test_host_on_agent_raises_on_already_hosted_comp(self):
        d = Distribution({'a1': ['v1', 'v2'], 'a2': ['v3']})
        self.assertRaises(ValueError, d.host_on_agent, 'a1', ['v3'])

    def test_host_on_new_agent(self):

        d = Distribution({'a1': ['v1', 'v2'], 'a2': ['v3']})
        d.host_on_agent('a3', ['v4'])


        self.assertEqual(d.agent_for('v4'), 'a3')

        self.assertIn('a3', d.agents)
        self.assertIn('v4', d.computations_hosted('a3'))

    def test_is_hosted_single_computation(self):
        d = Distribution({'a1': ['v1'], 'a2': ['v2']})
        self.assertTrue(d.is_hosted('v1'))
        self.assertTrue(d.is_hosted('v2'))
        self.assertFalse(d.is_hosted('v3'))

    def test_is_hosted_several_computation(self):
        d = Distribution({'a1': ['v1'], 'a2': ['v2']})

        self.assertTrue(d.is_hosted( ['v1', 'v2']))
        self.assertTrue(d.is_hosted(['v2', 'v1']))

        self.assertFalse(d.is_hosted(['v3']))
        self.assertFalse(d.is_hosted(['v1', 'v3']))
        self.assertFalse(d.is_hosted(['v3', 'v2']))


class TestDistributionHints(unittest.TestCase):

    def test_must_host(self):

        dh = DistributionHints(must_host={'a1': ['v1']})
        self.assertIn('v1', dh.must_host('a1'))

    def test_must_host_return_empty_when_not_specified(self):

        dh = DistributionHints(must_host={'a1': ['v1']})
        self.assertEqual(len(dh.must_host('a2')), 0)

    def test_host_with(self):
        dh = DistributionHints(host_with={'c1': ['v1']})

        self.assertIn('c1', dh.host_with('v1'))
        self.assertIn('v1', dh.host_with('c1'))

    def test_host_with_several(self):
        dh = DistributionHints(host_with={'c1': ['v1', 'v2']})

        self.assertIn('c1', dh.host_with('v1'))
        self.assertIn('v1', dh.host_with('c1'))
        self.assertIn('v2', dh.host_with('c1'))
        self.assertIn('v2', dh.host_with('c1'))
