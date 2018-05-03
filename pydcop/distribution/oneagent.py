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


from typing import List, Dict, Iterable, Callable
from collections import defaultdict

from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution, DistributionHints, \
    ImpossibleDistributionException


def distribution_cost(distribution: Distribution,
                      computation_graph: ComputationGraph,
                      agentsdef: Iterable[AgentDef],
                      computation_memory: Callable[[ComputationNode], float],
                      communication_load: Callable[[ComputationNode, str],
                                                   float]) -> float:
    return 0

def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory=None,
               communication_load=None)-> Distribution:
    """
    Simplistic distribution method: each computation is hosted on agent 
    agent and each agent host a single computation.
    Agent capacity is not considered.

    Raises an ImpossibleDistributionException
    
    :param computation_graph: 
    :param agentsdef: AgntsDef object containing the list of agent, there must
    be at least as many agents as computations
    :param hints DistributionHints
     
    :return: a distribution a dict {agent_name: [ var_name, ...]} 
    """

    agents = list(agentsdef)

    if len(agents) < len(computation_graph.nodes):
        raise ImpossibleDistributionException(
            'Not enough agents for one agent for each computation : {} < {}'
                .format(len(agents),len(computation_graph.nodes)))

    agent_names = [a.name for a in agents]
    distribution = defaultdict(lambda : list())
    for n, a in zip(computation_graph.nodes, agent_names):
        distribution[a].append(n.name)

    return Distribution(distribution)