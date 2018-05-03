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


from typing import List, Dict, Tuple

from pydcop.computations_graph.objects import ComputationGraph
from pydcop.infrastructure.discovery import Discovery


def _removal_orphaned_computations(
        departed: List[str], discovery: Discovery) -> List[str]:
    """
    Build the list of computation orphaned when removing some agents.

    Parameters
    ----------
    departed: list of str
        list of agent's names

    discovery: a Discovery instance
        information about the current deployment of computations

    Returns
    -------

    """
    orphaned = []
    for agt in departed:
        orphaned += discovery.agent_computations(agt)
    return orphaned


def _removal_candidate_agents(departed: List[str],
                              discovery: Discovery) \
        -> List[str]:
    """

    :param departed: a list of agents
    :param discovery

    :return: the candidate agents as a list of agents involved in the
    reparation process, i.e. candidates that could host one the the orphaned
    computation from the departed agents

    """
    orphaned = _removal_orphaned_computations(departed, discovery)
    candidate_agents = []

    for o in orphaned:
        candidate_agents += list(discovery.replica_agents(o))
    candidate_agents = list(set(candidate_agents).difference(set(departed)))

    return candidate_agents


def _removal_candidate_computations_for_agt(agt, orphaned_computations,
                                            discovery: Discovery):
    """

    :param agt:
    :param orphaned_computations:

    :return: The list of orphaned computations that could potentially be
    hosted on agt (because agt has their replica)
    """
    comps = []
    for o in orphaned_computations:
        if agt in discovery.replica_agents(o):
            comps.append(o)
    return comps


def _removal_candidate_computation_info(
        orphan: str, departed: List[str], cg: ComputationGraph,
        discovery: Discovery) \
        -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    All info needed by an agent to participate in negotiation about hosting
    the computation `comp`

    :param orphan: the candidate computation that must be hosted
    :param departed: the agent that left the system
    :param cg: the computation graph
    :param discovery: the distribution of computation on agents

    :return: a triple ( candidate_agents, fixed_neighbors, candidates_neighbors)
    where:

    * candidate agents is a list of agents that could host this computation
    * fixed_neighbors is a map comp->agent that indicates, for each
      neighbor computation of `comp` that is not a candidate (orphaned),
      its host agent
    * candidates_neighbors is a map comp -> List[agt] indicating which agent
      could host each of the neighbor computation that is also a candidate
      computation.

    """
    orphaned_computation = _removal_orphaned_computations(
        departed, discovery)

    candidate_agents = list(discovery.replica_agents(orphan).difference(
        departed))
    fixed_neighbors = {}
    candidates_neighbors = {}
    for n in cg.neighbors(orphan):
        if n == orphan:
            continue
        if n in orphaned_computation:
            candidates_neighbors[n] = \
                list(discovery.replica_agents(n).difference(departed))
        else:
            fixed_neighbors[n] = discovery.computation_agent(n)

    return candidate_agents, fixed_neighbors, candidates_neighbors


def _removal_candidate_agt_info(agt: str, departed: List[str],
                                cg: ComputationGraph,
                                discovery: Discovery) \
        -> Dict[str, Tuple[List[str], Dict[str, str], Dict[str, List[str]]]]:
    """
    :return: for a candidate agent, the full information needed to
    instantiate the decision dcop about hosting each of the orphaned
    computation that could be hosted on this agent.

    A Dict {computation -> _removal_candidate_computation_info}

    for each orphaned computation that could be hosted on this agent there is
    an entry in the return dict with a tuple generated by
    _removal_candidate_computation_info.

    see _removal_candidate_computation_info
    """
    info = {}
    orphaned = _removal_orphaned_computations(departed, discovery)
    for c in _removal_candidate_computations_for_agt(agt, orphaned, discovery):
        info[c] = _removal_candidate_computation_info(
            c, departed, cg, discovery)
    return info
