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

from pydcop.algorithms.objects import AlgoDef, load_algorithm_module, \
    list_available_algorithms
from pydcop.utils.simple_repr import simple_repr, from_repr


class AlgoDefTest(unittest.TestCase):

    def test_algo_def(self):

        a = AlgoDef('maxsum', 'min', {'stability': 0.01})

        self.assertEqual(a.algo, 'maxsum')
        self.assertEqual(a.mode, 'min')
        self.assertIn('stability', a.param_names())
        self.assertEqual(a.param_value('stability'), 0.01)

    def test_simple_repr(self):

        a = AlgoDef('maxsum', 'min', {'stability': 0.01})

        r = simple_repr(a)

        self.assertEqual(r['algo'], 'maxsum')
        self.assertEqual(r['mode'], 'min')
        self.assertEqual(r['params']['stability'], 0.01)

    def test_from_repr(self):

        a = AlgoDef('maxsum', 'min', {'stability': 0.01})

        r = simple_repr(a)
        a2 = from_repr(r)

        self.assertEqual(a, a2)
        self.assertEqual(a2.param_value('stability'), 0.01)


def test_building_algodef_with_default_params():

    a = AlgoDef.build_with_default_param('maxsum')

    assert a.params['damping'] == 0


def test_load_algorithm():

    # We test load for all available algorithms
    for a in list_available_algorithms():
        algo = load_algorithm_module(a)

        assert algo.algorithm_name == a
        assert hasattr(algo, 'communication_load')
        assert hasattr(algo, 'computation_memory')


def test_load_algorithm_with_default_footprint():

    # dsatuto has no load method defined : check that we get instead default
    # implementations
    algo = load_algorithm_module('dsatuto')
    assert algo.algorithm_name == 'dsatuto'
    assert algo.communication_load(None, None) == 1
    assert algo.computation_memory(None) == 1
