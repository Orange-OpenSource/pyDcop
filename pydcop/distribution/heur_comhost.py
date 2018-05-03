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
Distribution heuristic based on hosting costs and communication cost.

Greedy algorithm:
We place first the computation with the highest footprint.

"""
import logging
from typing import Iterable, Callable, List, Dict, Tuple

from collections import defaultdict

from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution import ilp_compref
from pydcop.distribution.objects import DistributionHints, Distribution

logger = logging.getLogger('distribution.heur_comhost')


# Weight factors when aggregating communication costs and hosting costs in the
# objective function.
# the global objective is built as Comm_cost * RATIO + Hosting_cost * (1-RATIO)
RATIO_HOST_COMM = 0.5


def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory: Callable[[ComputationNode], float]=None,
               communication_load: Callable[[ComputationNode, str],
                                            float]=None) \
        -> Distribution:
    """

    Parameters
    ----------
    computation_graph
    agentsdef
    hints
    computation_memory
    communication_load

    Returns
    -------

    """
    computations = sorted([(computation_memory(n), n, None)
                           for n in computation_graph.nodes],
                          key=lambda o: (o[0], o[1].name),
                          reverse=True)
    logger.info('placing computations %s',
                [(f, c.name) for f, c, _ in computations])

    current_mapping = {}  # Type: Dict[str, str]
    i = 0
    while len(current_mapping) != len(computations):
        footprint, computation, candidates = computations[i]
        logger.debug('Trying to place computation %s with footprint %s',
                     computation.name, footprint)
        # try
        # look for agent for computation c
        if candidates is None:
            candidates = candidate_hosts(computation, footprint,
                                         computations, agentsdef,
                                         communication_load, current_mapping)
            computations[i] = footprint, computation, candidates
        logger.debug('Candidates for computation %s : %s',
                     computation.name, candidates   )

        if not candidates:
            if i==0:
                raise ValueError('Impossible Distribution !')

            # no candidate : backtrack !
            i -= 1
            logger.info('No candidate for %s, backtrack placement '
                        'of computation %s (was on %s',
                        computation.name, computations[i][1].name,
                        current_mapping[computations[i][1].name])
            current_mapping.pop(computations[i][1].name)

            # FIXME : eliminate selected agent for previous computation
        else:
            _, selected = candidates.pop()
            current_mapping[computation.name] = selected.name
            computations[i] = footprint, computation, candidates
            logger.debug('Place computation %s on agent %s', computation.name,
                        selected.name)
            i += 1

    # Build the distribution for the mapping
    agt_mapping = defaultdict(lambda: [])
    for c, a in current_mapping.items():
        agt_mapping[a].append(c)
    dist = Distribution(agt_mapping)

    return dist


def distribution_cost(distribution: Distribution,
                      computation_graph: ComputationGraph,
                      agentsdef: Iterable[AgentDef],
                      computation_memory: Callable[[ComputationNode], float],
                      communication_load: Callable[[ComputationNode, str],
                                                   float]) -> float:
    return ilp_compref.distribution_cost(
        distribution, computation_graph, agentsdef,
        computation_memory, communication_load)


def candidate_hosts(computation: ComputationNode, footprint: float,
                    computations: List[Tuple],
                    agents: Iterable[AgentDef],
                    communication_load: Callable[[ComputationNode, str], float],
                    mapping: Dict[str, str]):
    candidates = []
    for agt in agents:
        # Compute remaining capacity for agt, to check if it as enough place
        # left. Only keep agents that have enough capacity.
        capa = agt.capacity
        for c, a in mapping.items():
            if a == agt.name:
                c_footprint = next(f for f, comp, _ in computations
                                 if comp.name == c)
                capa -= c_footprint
        if capa < footprint:
            continue

        # compute cost of assigning computation to agt
        hosting_cost = agt.hosting_cost(computation.name)
        comm_cost = 0
        for l in computation.links:
            for n in l.nodes:
                if n in mapping:
                    comm_cost += communication_load(computation, n) \
                            * agt.route(mapping[n])
        cost = RATIO_HOST_COMM * comm_cost + (1-RATIO_HOST_COMM) *hosting_cost
        candidates.append((cost, agt))

    candidates.sort(key=lambda o: (o[0], o[1].name), reverse=True)
    return candidates
