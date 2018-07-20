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

from pydcop.dcop.objects import Variable
from pydcop.algorithms import mgm
from pydcop.computations_graph.constraints_hypergraph \
    import VariableComputationNode
from pydcop.dcop.relations import constraint_from_str


def test_communication_load():
    v = Variable('v1', list(range(10)))
    var_node = VariableComputationNode(v, [])
    assert mgm.UNIT_SIZE + mgm.HEADER_SIZE \
           == mgm.communication_load(var_node, 'f1')


def test_computation_memory_one_constraint():
    v1 = Variable('v1', list(range(10)))
    v2 = Variable('v2', list(range(10)))
    v3 = Variable('v3', list(range(10)))
    c1 = constraint_from_str('c1', ' v1 + v2 == v3', [v1, v2, v3])
    v1_node = VariableComputationNode(v1, [c1])

    # here, we have an hyper-edges with 3 vertices
    assert mgm.computation_memory(v1_node) == mgm.UNIT_SIZE * 2


def test_computation_memory_two_constraints():
    v1 = Variable('v1', list(range(10)))
    v2 = Variable('v2', list(range(10)))
    v3 = Variable('v3', list(range(10)))
    v4 = Variable('v4', list(range(10)))
    c1 = constraint_from_str('c1', ' v1 == v2', [v1, v2])
    c2 = constraint_from_str('c2', ' v1 == v3', [v1, v3])
    c3 = constraint_from_str('c3', ' v1 == v4', [v1, v4])
    v1_node = VariableComputationNode(v1, [c1, c2, c3])

    # here, we have 3 edges , one for each constraint
    assert mgm.computation_memory(v1_node) == mgm.UNIT_SIZE * 3
