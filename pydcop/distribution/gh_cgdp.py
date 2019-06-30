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
GH-CGDP : Greedy Heuristic for Computation Graph distribution

Distribution heuristic based on hosting costs and communication cost,
while respecting agents' capacity.

Greedy algorithm:
* place computation with highest footprint first
* select agent with enough capacity left and smallest aggregated hosting
  and communication costs, according to the communication that have already been placed.


This means that the first computations (with biggest footprint) are mostly placed based
on their hosting cost, as most other computations are not distributed yet and we cannot
evaluate their communication cost.


"""
import logging
import random
from collections import defaultdict
from typing import Iterable, Callable, Tuple, List, Dict

from pydcop.computations_graph.objects import ComputationNode, ComputationGraph
from pydcop.dcop.objects import AgentDef
from pydcop.distribution import oilp_cgdp
from pydcop.distribution.objects import Distribution, ImpossibleDistributionException

logger = logging.getLogger("distribution.gh_cgdp")


# Weight factors when aggregating communication costs and hosting costs in the
# objective function.
# the global objective is built as Comm_cost * RATIO + Hosting_cost * (1-RATIO)
RATIO_HOST_COMM = oilp_cgdp.RATIO_HOST_COMM
# small: comm count less


def distribute(
    computation_graph: ComputationGraph,
    agentsdef: Iterable[AgentDef],
    hints=None,
    computation_memory: Callable[[ComputationNode], float] = None,
    communication_load: Callable[[ComputationNode, str], float] = None,
    timeout=None,  # not used
) -> Distribution:
    """
    gh-cgdp distribution method.

    Heuristic distribution baed on communication and hosting costs, while respecting
    agent's capacities

    Parameters
    ----------
    computation_graph
    agentsdef
    hints
    computation_memory
    communication_load

    Returns
    -------
    Distribution:
        The distribution for the computation graph.

    """

    # Place computations with hosting costs == 0
    # For SECP, this assign actuators var and factor to the right device.
    fixed_mapping = {}
    for comp in computation_graph.node_names():
        for agent in agentsdef:
            if agent.hosting_cost(comp) == 0:
                fixed_mapping[comp] = (
                    agent.name,
                    computation_memory(computation_graph.computation(comp)),
                )
                break

    # Sort computation by footprint, but add a random element to avoid sorting on names
    computations = [
        (computation_memory(n), n, None, random.random())
        for n in computation_graph.nodes
        if n.name not in fixed_mapping
    ]
    computations = sorted(computations, key=lambda o: (o[0], o[3]), reverse=True)
    computations = [t[:-1] for t in computations]
    logger.info("placing computations %s", [(f, c.name) for f, c, _ in computations])

    current_mapping = {}  # Type: Dict[str, str]
    i = 0
    while len(current_mapping) != len(computations):
        footprint, computation, candidates = computations[i]
        logger.debug(
            "Trying to place computation %s with footprint %s",
            computation.name,
            footprint,
        )
        # look for cancidiate agents for computation c
        # TODO: keep a list of remaining capacities for agents ?
        if candidates is None:
            candidates = candidate_hosts(
                computation,
                footprint,
                computations,
                agentsdef,
                communication_load,
                current_mapping,
                fixed_mapping,
            )
            computations[i] = footprint, computation, candidates
        logger.debug("Candidates for computation %s : %s", computation.name, candidates)

        if not candidates:
            if i == 0:
                logger.error(
                    f"Cannot find a distribution, no candidate for computation {computation}\n"
                    f" current mapping: {current_mapping}"
                )
                raise ImpossibleDistributionException(
                    f"Impossible Distribution, no candidate for {computation}"
                )

            # no candidate : backtrack !
            i -= 1
            logger.info(
                "No candidate for %s, backtrack placement "
                "of computation %s (was on %s",
                computation.name,
                computations[i][1].name,
                current_mapping[computations[i][1].name],
            )
            current_mapping.pop(computations[i][1].name)

            # FIXME : eliminate selected agent for previous computation
        else:
            _, selected = candidates.pop()
            current_mapping[computation.name] = selected.name
            computations[i] = footprint, computation, candidates
            logger.debug(
                "Place computation %s on agent %s", computation.name, selected.name
            )
            i += 1

    # Build the distribution for the mapping
    agt_mapping = defaultdict(lambda: [])
    for c, a in current_mapping.items():
        agt_mapping[a].append(c)
    for c, (a, _) in fixed_mapping.items():
        agt_mapping[a].append(c)
    dist = Distribution(agt_mapping)

    return dist


def distribution_cost(
    distribution: Distribution,
    computation_graph: ComputationGraph,
    agentsdef: Iterable[AgentDef],
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
) -> float:
    return oilp_cgdp.distribution_cost(
        distribution,
        computation_graph,
        agentsdef,
        computation_memory,
        communication_load,
    )


def candidate_hosts(
    computation: ComputationNode,
    footprint: float,
    computations: List[Tuple],
    agents: Iterable[AgentDef],
    communication_load: Callable[[ComputationNode, str], float],
    mapping: Dict[str, str],
    fixed_mapping: Dict[str, Tuple[str, float]],
):
    """
    Build a list of candidate agents for a computation.

    The list includes agents that have enough capacity to host this computation
    and is sorted by cost (cheapest cost) where cost is the aggregated
    hosting and communication cost incurred by hosting the computation on that agent,
    according to the computation that have been already distributed.
    This means that the first computations are mostly placed depending on their
    hosting cost, as most other computations are not distributed yet and we cannot
    evaluate their communication cost.

    Parameters
    ----------
    computation
    footprint
    computations
    agents
    communication_load
    mapping

    Returns
    -------

    """
    candidates = []
    for agt in agents:
        # Compute remaining capacity for agt, to check if it as enough place
        # left. Only keep agents that have enough capacity.
        capa = agt.capacity
        for c, a in mapping.items():
            if a == agt.name:
                c_footprint = next(f for f, comp, _ in computations if comp.name == c)
                capa -= c_footprint
        for c, (a, f) in fixed_mapping.items():
            if a == agt.name:
                capa -= f

        if capa < footprint:
            continue

        # compute cost of assigning computation to agt
        hosting_cost = agt.hosting_cost(computation.name)
        comm_cost = 0
        for l in computation.links:
            for n in l.nodes:
                if n in mapping:
                    comm_cost += communication_load(computation, n) * agt.route(
                        mapping[n]
                    )
        cost = RATIO_HOST_COMM * comm_cost + (1 - RATIO_HOST_COMM) * hosting_cost
        candidates.append((cost, agt))

    # Avoid sorting ties by name by adding a random element in the tuple.
    # Otherwise, when agents have the same capacity, agents with names sorted first
    # will always get more computations.
    candidates = [(c, a, random.random()) for c, a in candidates]
    candidates.sort(key=lambda o: (o[0], o[2]), reverse=True)
    candidates = [t[:-1] for t in candidates]

    return candidates
