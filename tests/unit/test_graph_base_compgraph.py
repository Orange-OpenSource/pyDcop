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


import pytest

from pydcop.computations_graph.objects import ComputationNode, Link
from pydcop.utils.simple_repr import from_repr, simple_repr


def test_node_creation_minimal():
    # name is the only mandatory param:
    n = ComputationNode('n1')
    assert n.name == 'n1'
    assert not n.type
    assert not n.links
    assert not n.neighbors

def test_node_creation_with_links():

    n1 = ComputationNode('n1', links=[Link(['n2'])])

    assert 'n2' in n1.neighbors
    assert list(n1.links)[0].has_node('n2')

def test_node_creation_with_hyperlinks():

    n1 = ComputationNode('n1', links=[Link(['n2', 'n3']),
                                      Link(['n4'])])

    assert 'n2' in n1.neighbors
    assert 'n3' in n1.neighbors
    assert 'n4' in n1.neighbors

def test_node_creation_with_one_neighbor():

    n1 = ComputationNode('n1', neighbors=['n2'])

    assert 'n2' in n1.neighbors
    assert len(n1.links) == 1
    assert list(n1.links)[0].has_node('n2')

def test_node_creation_with_several_neighbors():

    n1 = ComputationNode('n1', neighbors=['n2', 'n3', 'n4'])

    assert 'n2' in n1.neighbors
    assert 'n3' in n1.neighbors
    assert 'n4' in n1.neighbors
    assert len(n1.links) == 3


def test_node_creation_raises_when_giving_links_neighbors():

    with pytest.raises(ValueError):
        n1 = ComputationNode('n1', links=[Link(['n2'])], neighbors=['n2'])


def test_node_simplerepr():
    n1 = ComputationNode('n1', neighbors=['n2', 'n3', 'n4'])

    r1 = simple_repr(n1)

    obtained = from_repr(r1)

    assert n1 == obtained
    assert 'n2' in n1.neighbors
    assert 'n3' in n1.neighbors
    assert 'n4' in n1.neighbors
    assert len(n1.links) == 3