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
OILP-SECP-CGDP : Optimal ILP-based SECP Computation Graph Distribution

Distribution of a computation graph for a SECP, when using a constraint graph based DCOP
algorithm

This distribution method only takes into account communication loads between computations,
not hosting nor route cost.

This distribution method is designed for SECP and makes the following assumptions on
the DCOP (which are always satisfied if the SECP has been generated using
'pydcop generate' command):
* variables that represent an actuator are assigned to an agent, with an hosting cost of 0
* all agent must have a capacity attribute



"""
import logging
import time
from itertools import combinations
from typing import Iterable, Callable, List
from collections import defaultdict

from pulp import (
    LpVariable,
    LpBinary,
    LpMinimize,
    LpProblem,
    lpSum,
    LpStatusOptimal,
    GLPK_CMD,
    pulp,
    LpAffineExpression,
)

from pydcop.computations_graph.constraints_hypergraph import (
    ComputationConstraintsHyperGraph,
)
from pydcop.computations_graph.objects import ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution, ImpossibleDistributionException
from pydcop.distribution.oilp_secp_fgdp import (
    secp_dist_objective_function,
    secp_computation_memory_in_cg,
)

logger = logging.getLogger("distribution.oilp_secp_cgdp")


def distribute(
    computation_graph: ComputationConstraintsHyperGraph,
    agentsdef: Iterable[AgentDef],
    hints=None,
    computation_memory: Callable[[ComputationNode], float] = None,
    communication_load: Callable[[ComputationNode, str], float] = None,
    timeout=600,  # Max 10 min
) -> Distribution:
    if computation_memory is None or communication_load is None:
        raise ImpossibleDistributionException(
            "oilp_secp_cgdp distribution requires "
            "computation_memory and link_communication functions"
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
                agents_capa[agent.name] -= computation_memory(
                    computation_graph.computation(comp)
                )
                if agents_capa[agent.name] < 0:
                    raise ImpossibleDistributionException(
                        f"Not enough capacity on {agent} to hosts actuator {comp}: {agents_capa[agent.name]}"
                    )

    logger.info(f"Actuator variables on agents: {dict(mapping)}")
    logger.info(f"Remaining capacity: {dict(agents_capa)}")

    return cg_secp_ilp(
        computation_graph,
        agentsdef,
        Distribution(mapping),
        computation_memory,
        communication_load,
    )




def distribution_cost(
    distribution: Distribution,
    computation_graph: ComputationConstraintsHyperGraph,
    agentsdef: Iterable[AgentDef],
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
) -> float:
    """
    Compute the cost of the distribution.

    Only takes communication costs into account (no hosting nor route costs).

    Parameters
    ----------
    distribution
    computation_graph
    agentsdef
    computation_memory
    communication_load

    Returns
    -------

    """
    comm = 0
    agt_names = [a.name for a in agentsdef]
    for l in computation_graph.links:
        # As we support hypergraph, we may have more than 2 ends to a link
        for c1, c2 in combinations(l.nodes, 2):
            if distribution.agent_for(c1) != distribution.agent_for(c2):
                edge_cost = communication_load(computation_graph.computation(c1), c2)
                logger.debug(f"edge cost between {c1} and {c2} :  {edge_cost}")
                comm += edge_cost
            else:
                logger.debug(f"On same agent, no edge cost between {c1} and {c2}")

    # This distribution model only takes communication cost into account.
    # cost = RATIO_HOST_COMM * comm + (1-RATIO_HOST_COMM) * hosting
    return comm, comm, 0


def cg_secp_ilp(
    cg: ComputationConstraintsHyperGraph,
    agents: List[AgentDef],
    already_assigned: Distribution,
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
    timeout=600,  # Max 10 min
) -> Distribution:
    start_t = time.time()

    agents = list(agents)
    agents_names = [a.name for a in agents]

    # Only keep computations for which we actually need to find an agent.
    comps_to_host = [
        c for c in cg.node_names() if not already_assigned.has_computation(c)
    ]

    # x_i^k : binary variable indicating if var x_i is hosted on agent a_k.
    xs = _build_cs_binvar(comps_to_host, agents_names)
    # alpha_ijk : binary variable indicating if  x_i and f_j are both on a_k.
    alphas = _build_alphaijk_binvars(cg, agents_names)
    logger.debug(f"alpha_ijk {alphas}")

    # LP problem with objective function (total communication cost).
    pb = LpProblem("distribution", LpMinimize)
    pb += (
        _objective_function(cg, communication_load, alphas, agents_names),
        "Communication costs",
    )

    # Constraints.
    # All variable computations must be hosted:
    for i in comps_to_host:
        pb += (
            lpSum([xs[(i, k)] for k in agents_names]) == 1,
            "var {} is hosted".format(i),
        )
    # Each agent must host at least one computation:
    # We only need this constraints for agents that do not already host a
    # computation:
    empty_agents = [
        a for a in agents_names if not already_assigned.computations_hosted(a)
    ]
    for k in empty_agents:
        pb += (
            lpSum([xs[(i, k)] for i in comps_to_host]) >= 1,
            "atleastone {}".format(k),
        )

    # Memory capacity constraint for agents
    for a in agents:
        # Decrease capacity for already hosted computations
        capacity = a.capacity - sum(
            [
                secp_computation_memory_in_cg(c, cg, computation_memory)
                for c in already_assigned.computations_hosted(a.name)
            ]
        )

        pb += (
            lpSum(
                [
                    secp_computation_memory_in_cg(i, cg, computation_memory)
                    * xs[(i, a.name)]
                    for i in comps_to_host
                ]
            )
            <= capacity,
            "memory {}".format(a.name),
        )

    # Linearization constraints for alpha_ijk.
    for (i, j), k in alphas:

        if i in comps_to_host and j in comps_to_host:
            pb += alphas[((i, j), k)] <= xs[(i, k)], "lin1 {}{}{}".format(i, j, k)
            pb += alphas[((i, j), k)] <= xs[(j, k)], "lin2 {}{}{}".format(i, j, k)
            pb += (
                alphas[((i, j), k)] >= xs[(i, k)] + xs[(j, k)] - 1,
                "lin3 {}{}{}".format(i, j, k),
            )

        elif i in comps_to_host and j not in comps_to_host:
            # Var is free, factor is already hosted
            if already_assigned.agent_for(j) == k:
                pb += alphas[((i, j), k)] == xs[(i, k)]
            else:
                pb += alphas[((i, j), k)] == 0

        elif i not in comps_to_host and j in comps_to_host:
            # if i is not in vars_vars_to_host, it means that it's a
            # computation that is already hosted (from  hints)
            if already_assigned.agent_for(i) == k:
                pb += alphas[((i, j), k)] == xs[(j, k)]
            else:
                pb += alphas[((i, j), k)] == 0

        else:
            # i and j are both alredy hosted
            if (
                already_assigned.agent_for(i) == k
                and already_assigned.agent_for(j) == k
            ):
                pb += alphas[((i, j), k)] == 1
            else:
                pb += alphas[((i, j), k)] == 0


    # the timeout for the solver must be monierd by the time spent to build the pb:
    remaining_time = round(timeout - (time.time() - start_t)) -2

    # Now solve our LP
    status = pb.solve(GLPK_CMD(keepFiles=0, msg=False, options=["--pcost",  "--tmlim", str(remaining_time)]))

    if status != LpStatusOptimal:
        raise ImpossibleDistributionException("No possible optimal" " distribution ")
    else:
        logger.debug("GLPK cost : %s", pulp.value(pb.objective))

        comp_dist = already_assigned
        for k in agents_names:

            agt_vars = [i for i, ka in xs if ka == k and pulp.value(xs[(i, ka)]) == 1]
            comp_dist.host_on_agent(k, agt_vars)

        return comp_dist


def _build_cs_binvar(comps_to_host, agents_names):
    if not comps_to_host:
        return {}
    return LpVariable.dict("x", (comps_to_host, agents_names), cat=LpBinary)


def _build_alphaijk_binvars(
    cg: ComputationConstraintsHyperGraph, agents_names: Iterable[str]
):
    # As these variables are only used in the objective function,
    # when optimizing communication cost, we only need them when (i,j) is an
    # edge in the factor graph
    edge_indexes = []
    for link in cg.links:
        for end1, end2 in combinations(link.nodes, 2):
            edge_indexes.append((end1, end2))

    alphas = LpVariable.dict("a", (edge_indexes, agents_names), cat=LpBinary)
    return alphas


def _objective_function(
    cg: ComputationConstraintsHyperGraph, communication_load, alphas, agents_names
):
    # The objective function is the negated sum of the communication cost on
    # the links in the constraint graph.

    objective = LpAffineExpression()

    for agt in agents_names:
        for link in cg.links:
            # logger.debug(f"link {link!r}")
            if len(link.nodes) == 1:
                # link representing a unary constraint: no com cost
                continue
            objective += lpSum(
                [
                    -communication_load(cg.computation(end1), end2)
                    * alphas[((end1, end2), agt)]
                    for end1, end2 in combinations(link.nodes, 2)
                    if ((end1, end2), agt) in alphas
                ]
            )

    logger.debug(f"Objective: {objective}")
    return objective
