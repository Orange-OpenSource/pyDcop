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

"""

The ``oneagent`` distribution algorithm assigns exactly one computation to each
agent in the system.

It is the most simple distribution and, when used with many DCOP algorithms,
it replicates the traditional hypothesis used in the DCOP literature
where each agent is responsible for exactly one variable.

Note that this applies to algorithms using a computation-hyper graph model,
like DSA, MGM, etc.

The ``oneagent`` distribution does not define any notion of distribution cost.


Functions
---------

.. autofunction:: pydcop.distribution.oneagent.distribute

.. autofunction:: pydcop.distribution.oneagent.distribution_cost


"""

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
    """
    As the ``oneagent`` distribution does not define any notion of
    distribution cost, this function always returns 0.

    Parameters
    ----------
    distribution
    computation_graph
    agentsdef
    computation_memory
    communication_load

    Returns
    -------
    distribution cost:
        0
    """
    return 0, 0, 0

def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory=None,
               communication_load=None,
               timeout= None)-> Distribution:
    """
    Simplistic distribution method: each computation is hosted on agent 
    agent and each agent host a single computation.
    Agent capacity is not considered.

    Raises an ImpossibleDistributionException

    Parameters
    ----------
    computation_graph: a ComputationGraph
         the computation graph containing the computation that must be
         distributed
    agentsdef: iterable of AgentDef objects
        The definition of the agents the computation will be assigned to.
        There **must** be at least as many agents as computations.
    hints:
        Not used by the ``oneagent`` distribution method.
    computation_memory:
        Not used by the ``oneagent`` distribution method.
    computation_memory:
        Not used by the ``oneagent`` distribution method.

    Returns
    -------
    distribution: Distribution
        A Distribution object containing the mapping form agents to
        computations.
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