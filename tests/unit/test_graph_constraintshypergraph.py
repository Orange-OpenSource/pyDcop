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

from pydcop.computations_graph.constraints_hypergraph import \
    VariableComputationNode, ConstraintLink, build_computation_graph
from pydcop.dcop.objects import Domain, Variable
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import constraint_from_str
from pydcop.utils.simple_repr import from_repr, simple_repr


def test_create_node_no_neigbors():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5', [v1])

    n1 = VariableComputationNode(v1, [c1])

    assert v1 == n1.variable
    assert c1 in n1.constraints
    assert len(n1.links) == 1  # link to our-self
    assert not n1.neighbors


def test_create_node_with_custom_name():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5', [v1])

    n1 = VariableComputationNode(v1, [c1], name='foo')

    assert v1 == n1.variable
    assert n1.name == 'foo'


def test_create_node_with_binary_constraint():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5 - v2', [v1, v2])

    n1 = VariableComputationNode(v1, [c1])

    assert v1 == n1.variable
    assert c1 in n1.constraints
    assert list(n1.links)[0].has_node('v2')
    assert 'v2' in n1.neighbors


def test_create_node_with_nary_constraint():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5 - v2 + v3', [v1, v2, v3])

    n1 = VariableComputationNode(v1, [c1])

    assert v1 == n1.variable
    assert c1 in n1.constraints
    assert len(list(n1.links)) ==1
    assert list(n1.links)[0].has_node('v2')
    assert list(n1.links)[0].has_node('v3')
    assert 'v2' in n1.neighbors
    assert 'v3' in n1.neighbors


def test_var_node_simple_repr():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5', [v1])

    cv1 = VariableComputationNode(v1, [c1])

    r = simple_repr(cv1)
    cv1_obtained = from_repr(r)

    assert cv1 == cv1_obtained


def test_link_simple_repr():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5 + v2 - v3', [v1, v2, v3])

    cv1 = VariableComputationNode(v1, [c1])
    cv2 = VariableComputationNode(v2, [c1])
    cv3 = VariableComputationNode(v3, [c1])

    link = ConstraintLink(c1.name, ['c1', 'c2', 'c3'])

    r = simple_repr(link)
    link_obtained = from_repr(r)

    assert link == link_obtained


def test_build_graph_from_dcop():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    dcop = DCOP('dcop_test', 'min')
    dcop += 'c1', 'v1 * 0.5 + v2 - v3', [v1, v2, v3]
    c1 = dcop.constraints['c1']

    graph = build_computation_graph(dcop)

    links = list(graph.links)
    assert 1 == len(links)
    assert ConstraintLink(c1.name, {'v1', 'v2', 'v3'}) in links

    nodes = list(graph.nodes)
    assert 3 == len(nodes)
    assert VariableComputationNode(v1, [c1]) in nodes
    assert VariableComputationNode(v2, [c1]) in nodes
    assert VariableComputationNode(v3, [c1]) in nodes


def test_build_graph_from_variables_constraints():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5 + v2 - v3', [v1, v2, v3])

    graph = build_computation_graph(variables=[v1, v2, v3],
                                    constraints=[c1])



def test_graph_density():
    d = Domain('d', 'test', [1, 2, 3])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    c1 = constraint_from_str('c1', 'v1 * 0.5 + v2 - v3', [v1, v2, v3])

    dcop = DCOP('dcop_test', 'min' )
    dcop.add_constraint(c1)

    graph = build_computation_graph(dcop)

    density = graph.density()
    assert density == 1/3