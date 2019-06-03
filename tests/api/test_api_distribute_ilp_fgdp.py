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


"""Api tests for ILP_FGPD distribution method.
"""
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, create_variables, create_agents
from tests.api.instances_and_utils import dcop_graphcoloring_3


def create_dcop():
    dcop = DCOP('test')
    # Domain and vraibales
    d = Domain('color', '', ['R', 'G'])
    variables = create_variables('v', [1, 2, 3], d)
    # unary constraints for preferences
    dcop += 'cost_1', '-0.1 if v1 == "R" else 0.1 ', variables
    dcop += 'cost_2', '-0.1 if v2 == "G" else 0.1 ', variables
    dcop += 'cost_3', '-0.1 if v3 == "G" else 0.1 ', variables
    # coloring constraints : v1 != v2 != v3
    dcop += 'c1', '1 if v1 == v2 else 0', variables
    dcop += 'c2', '1 if v3 == v2 else 0', variables

    return dcop


def test_api_distribute_maxsum_ilp_fgdp():
    from pydcop.computations_graph import factor_graph
    from pydcop.distribution import ilp_fgdp
    from pydcop.algorithms import amaxsum

    dcop = dcop_graphcoloring_3()
    agents = create_agents('a', range(1, 4), capacity=100)
    dcop._agents_def = agents

    cg = factor_graph.build_computation_graph(dcop)
    dist = ilp_fgdp.distribute(cg, dcop.agents.values(),
                               computation_memory=amaxsum.computation_memory,
                               communication_load=amaxsum.communication_load)

    assert dist.is_hosted(['v1', 'v2', 'v3'])


def test_api_distribute_dsa_ilp_fgdp():
    from pydcop.computations_graph import factor_graph
    from pydcop.distribution import ilp_fgdp
    from pydcop.algorithms import dsa

    dcop = dcop_graphcoloring_3()
    agents = create_agents('a', range(1, 4), capacity=100)
    dcop._agents_def = agents

    cg = factor_graph.build_computation_graph(dcop)
    dist = ilp_fgdp.distribute(cg, dcop.agents.values(),
                               computation_memory=dsa.computation_memory,
                               communication_load=dsa.communication_load)

    assert dist.is_hosted(['v1', 'v2', 'v3'])
