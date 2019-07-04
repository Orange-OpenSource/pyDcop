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

GH-SECP-CGDP : Greedy Heuristic for the SECP Constraint Graph Distribution Problem

In order to minimize communication load, we assign the computations for each yj to one of the
agents already hosting a variable that shares a constraint with yj . This of course only holds if the
algorithm supports n-ary constraints and we do not need to apply any binarization methods, which
would introduce additional auxiliary variables that must be distributed on agents.

When implementing this heuristic, we use a greedy approach to select
one agent among the set of valid agents for a given computation: we select the agent, with enough
capacity, that is already hosting the highest number of computations that share a dependency with
the computation we are placing. In case of tie, we chose the agent with the highest remaining
capacity. By grouping interdependent computations, this approach favors distributions with a low
communication cost.

Notice that
* the communication load is not used in this distribution method
* the computation footprint is required


This distribution method is designed for SECP and makes the following assumptions on
the DCOP:
* variables that represent an actuator are assigned to an agent, with an hosting cost of 0


"""
import logging
from typing import Iterable, Callable, List, Dict
from collections import defaultdict

from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution import oilp_secp_fgdp, oilp_secp_cgdp
from pydcop.distribution.objects import (
    DistributionHints,
    Distribution,
    ImpossibleDistributionException,
)

logger = logging.getLogger("distribution.gh_secp_cgdp")


def distribute(
    computation_graph: ComputationGraph,
    agentsdef: Iterable[AgentDef],
    hints: DistributionHints = None,
    computation_memory: Callable[[ComputationNode], float] = None,
    communication_load: Callable[[ComputationNode, str], float] = None,
    timeout=None,  # not used
) -> Distribution:
    if computation_memory is None:
        raise ImpossibleDistributionException(
            "adhoc distribution requires " "computation_memory functions"
        )

    mapping = defaultdict(lambda: [])
    agents_capa = {a.name: a.capacity for a in agentsdef}
    computations = computation_graph.node_names()
    # as we're dealing with a secp modelled as a constraint graph,
    # we only have actuator and pysical model variables.

    # First, put each actuator variable on its agent
    for agent in agentsdef:
        for comp in computation_graph.node_names():
            if agent.hosting_cost(comp) == 0:
                mapping[agent.name].append(comp)
                computations.remove(comp)
                agents_capa[agent.name] -= computation_memory(computation_graph.computation(comp))
                if agents_capa[agent.name] < 0:
                    raise ImpossibleDistributionException(
                        f"Not enough capacity on {agent} to hosts actuator {comp}: {agents_capa[agent.name]}"
                    )
                break
    logger.info(f"Actuator variables on agents: {dict(mapping)}")

    # We must now place physical model variable on an agent that host
    # a variable it depends on.
    # As physical models always depends on actuator variable,
    # there must always be a computation it depends on that is already hosted.

    for comp in computations:
        footprint = computation_memory(computation_graph.computation(comp))
        neighbors = computation_graph.neighbors(comp)

        candidates = find_candidates(agents_capa, comp, footprint, mapping, neighbors)

        # Host the computation on the first agent and decrease its remaining capacity
        selected = candidates[0][2]
        mapping[selected].append(comp)
        agents_capa[selected] -= footprint

    return Distribution({a: list(mapping[a]) for a in mapping})


def distribution_cost(
    distribution: Distribution,
    computation_graph: ComputationGraph,
    agentsdef: Iterable[AgentDef],
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
) -> float:
    return oilp_secp_cgdp.distribution_cost(
        distribution,
        computation_graph,
        agentsdef,
        computation_memory,
        communication_load,
    )

def find_candidates(agents_capa: Dict[str,int], comp: str, footprint: float, mapping: Dict, neighbors: Iterable[str]):
    # Candidate : agents with enough capacity, that host at least
    # one neighbor computation
    candidates = []
    for agent, capa in agents_capa.items():
        hosted_neighbors = len(set(mapping[agent]).intersection(neighbors))

        # logger.debug(
        #     f"agent: {agent} {hosted_neighbors} {set(mapping[agent])} - {neighbors}")
        hosted_neighbors = len(set(mapping[agent]).intersection(neighbors))
        if hosted_neighbors > 0 and capa >= footprint:
            candidates.append((hosted_neighbors, capa, agent))
    if not candidates:
        logger.error(f"Cannot host {comp} with footprint {footprint} - no valid candidate")

        logger.error(f"Already hosted: {dict(mapping)}")
        logger.error(f"Remaining agents capacity: {agents_capa} - {footprint}")

        raise ImpossibleDistributionException(
            f"No neighbor or not enough capacity to host {comp}")

    # Now, sort candidate agent by the number of neighbor computation they host
    # and remaining capacity
    candidates.sort(reverse=True)
    return candidates
