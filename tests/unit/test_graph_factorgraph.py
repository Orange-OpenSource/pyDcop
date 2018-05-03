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

from pydcop.computations_graph.factor_graph import ComputationsFactorGraph, \
    VariableComputationNode, FactorComputationNode, FactorGraphLink
from pydcop.computations_graph.factor_graph import build_computation_graph
from pydcop.dcop.objects import Variable, Domain
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import constraint_from_str
from pydcop.utils.simple_repr import simple_repr, from_repr


def test_one_var_one_factor():
    dcop = DCOP('test', 'min')
    d1 = Domain('d1', '--', [1, 2, 3])
    v1 = Variable('v1', d1)
    dcop += 'c1', '0.5 * v1', [v1]

    g = build_computation_graph(dcop)

    assert len(g.links) == 1
    assert len(g.nodes) == 2


def test_two_var_one_factor():
    dcop = DCOP('test', 'min')
    d1 = Domain('d1', '--', [1, 2, 3])
    v1 = Variable('v1', d1)
    v2 = Variable('v2', d1)
    dcop += 'c1', '0.5 * v1 + v2', [v1, v2]

    g = build_computation_graph(dcop)

    assert len(g.links) == 2
    assert len(g.nodes) == 3


def test_density_two_var_one_factor():
    dcop = DCOP('test', 'min')
    d1 = Domain('d1', '--', [1, 2, 3])
    v1 = Variable('v1', d1)
    v2 = Variable('v2', d1)
    dcop += 'c1', '0.5 * v1 + v2', [v1, v2]

    g = build_computation_graph(dcop)

    assert g.density() == 4/6


class TestFactorGraphComputation(unittest.TestCase):

    # Test computation & nodes

    def test_create_ok(self):
        d1 = Domain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        f1 = constraint_from_str('f1', 'v1 * 0.5', [v1])

        cv1 = VariableComputationNode(v1, [f1])
        cf1 = FactorComputationNode(f1)
        cg = ComputationsFactorGraph([cv1], [cf1])

    def test_raise_when_duplicate_computation_name(self):
        d1 = Domain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        # here we create a relation with the same name as the variable
        f1 = constraint_from_str('v1', 'v1 * 0.5', [v1])

        cv1 = VariableComputationNode(v1, ['f1'])
        cf1 = FactorComputationNode(f1)
        self.assertRaises(KeyError, ComputationsFactorGraph, [cv1],
                          [cf1])



def test_factornode_simple_repr():
    d1 = Domain('d1', '', [1, 2, 3, 5])
    v1 = Variable('v1', d1)
    f1 = constraint_from_str('f1', 'v1 * 0.5', [v1])

    cv1 = VariableComputationNode(v1, ['f1'])
    cf1 = FactorComputationNode(f1, )

    r= simple_repr(cf1)
    obtained = from_repr(r)

    assert obtained == cf1
    assert cf1.factor == obtained.factor


def test_variablenode_simple_repr():
    d1 = Domain('d1', '', [1, 2, 3, 5])
    v1 = Variable('v1', d1)
    f1 = constraint_from_str('f1', 'v1 * 0.5', [v1])

    cv1 = VariableComputationNode(v1, ['f1'])
    cf1 = FactorComputationNode(f1, )

    r= simple_repr(cv1)
    obtained = from_repr(r)

    assert obtained == cv1
    assert cv1.variable == obtained.variable


