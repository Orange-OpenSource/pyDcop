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
OILP-SECP-FGDP

This distribution method uses an ILP to compute an optimal distribution,
optimizing for communication costs, while respecting agents' capacities
and hosting at least one computation on each agent.

This distribution method is designed for SECP and makes the following assumptions on
the DCOP:
* variables that represent an actuator are assigned to an agent, with an hosting cost of 0

"""
import logging
import time
from collections import defaultdict
from itertools import combinations
from typing import Iterable, List, Dict, Callable

from pulp import (
    LpVariable,
    LpBinary,
    LpMinimize,
    lpSum,
    LpProblem,
    GLPK_CMD,
    LpStatusOptimal,
    pulp,
)

from pydcop.computations_graph.factor_graph import (
    ComputationsFactorGraph,
    VariableComputationNode,
    FactorComputationNode,
)
from pydcop.computations_graph.objects import ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import ImpossibleDistributionException, Distribution

logger = logging.getLogger("distribution.oilp_secp_fgdp")


def distribute(
    computation_graph: ComputationsFactorGraph,
    agentsdef: Iterable[AgentDef],
    hints=None,
    computation_memory: Callable[[ComputationNode], float] = None,
    communication_load: Callable[[ComputationNode, str], float] = None,
    timeout=600,  # Max 10 min
) -> Distribution:
    if computation_memory is None or communication_load is None:
        raise ImpossibleDistributionException(
            "oilp_secp_fgdp distribution requires "
            "computation_memory and link_communication functions"
        )

    mapping = defaultdict(lambda: [])
    agents_capa = {a.name: a.capacity for a in agentsdef}
    variable_computations, factor_computations = [], []
    for comp in computation_graph.nodes:
        if isinstance(comp, VariableComputationNode):
            variable_computations.append(comp.name)
        elif isinstance(comp, FactorComputationNode):
            factor_computations.append(comp.name)
        else:
            raise ImpossibleDistributionException(
                f"Error: {comp} is neither a factor nor a variable computation"
            )
    # actuators variables and cost factor on the corresponding agent:
    for variable in variable_computations[:]:

        for agent in agentsdef:
            if agent.hosting_cost(variable) == 0:
                # Found an actuator variable, host it on the agent
                mapping[agent.name].append(variable)
                variable_computations.remove(variable)
                agents_capa[agent.name] -= computation_memory(
                    computation_graph.computation(variable)
                )
                # search for the cost factor, if any, and host it on the same agent.
                for factor in factor_computations[:]:
                    if f"c_{variable}" == factor:
                        mapping[agent.name].append(factor)
                        factor_computations.remove(factor)
                        agents_capa[agent.name] -= computation_memory(
                            computation_graph.computation(factor)
                        )
                if agents_capa[agent.name] < 0:
                    raise ImpossibleDistributionException(
                        f"Not enough capacity on {agent} to hosts actuator {variable}: {agents_capa[agent.name]}"
                    )
                break
    logger.info(f"Actuator variables - agents: {dict(mapping)}")
    logger.info(f"Remaining capacity: {dict(agents_capa)}")

    return fg_secp_ilp(
        computation_graph,
        agentsdef,
        Distribution(mapping),
        computation_memory,
        communication_load,
    )


def distribution_cost(
    distribution: Distribution,
    computation_graph: ComputationsFactorGraph,
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


def fg_secp_ilp(
    cg: ComputationsFactorGraph,
    agents: List[AgentDef],
    already_assigned: Distribution,
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
    timeout=600,  # Max 10 min
) -> Distribution:
    start_t = time.time()

    variables = [n for n in cg.nodes if n.type == "VariableComputation"]
    factors = [n for n in cg.nodes if n.type == "FactorComputation"]

    agents = list(agents)
    agents_names = [a.name for a in agents]

    # Only keep computations for which we actually need to find an agent.
    vars_to_host = [
        v.name for v in variables if not already_assigned.has_computation(v.name)
    ]
    facs_to_host = [
        f.name for f in factors if not already_assigned.has_computation(f.name)
    ]

    # x_i^k : binary variable indicating if var x_i is hosted on agent a_k.
    xs = _build_xs_binvar(vars_to_host, agents_names)
    # f_j^k : binary variable indicating if factor f_j is hosted on agent a_k.
    fs = _build_fs_binvar(facs_to_host, agents_names)
    # alpha_ijk : binary variable indicating if  x_i and f_j are both on a_k.
    alphas = _build_alphaijk_binvars(cg, agents_names)
    logger.debug(f"alpha_ijk {alphas}")

    # LP problem with objective function (total communication cost).
    pb = LpProblem("distribution", LpMinimize)
    pb += (
        secp_dist_objective_function(cg, communication_load, alphas, agents_names),
        "Communication costs",
    )

    # Constraints.
    # All variable computations must be hosted:
    for i in vars_to_host:
        pb += (
            lpSum([xs[(i, k)] for k in agents_names]) == 1,
            "var {} is hosted".format(i),
        )

    # All factor computations must be hosted:
    for j in facs_to_host:
        pb += (
            lpSum([fs[(j, k)] for k in agents_names]) == 1,
            "factor {} is hosted".format(j),
        )

    # Each agent must host at least one computation:
    # We only need this constraints for agents that do not already host a
    # computation:
    empty_agents = [
        a for a in agents_names if not already_assigned.computations_hosted(a)
    ]
    for k in empty_agents:
        pb += (
            lpSum([xs[(i, k)] for i in vars_to_host])
            + lpSum([fs[(j, k)] for j in facs_to_host])
            >= 1,
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
                    for i in vars_to_host
                ]
            )
            + lpSum(
                [
                    secp_computation_memory_in_cg(j, cg, computation_memory)
                    * fs[(j, a.name)]
                    for j in facs_to_host
                ]
            )
            <= capacity,
            "memory {}".format(a.name),
        )

    # Linearization constraints for alpha_ijk.
    for link in cg.links:
        i, j = link.variable_node, link.factor_node
        for k in agents_names:

            if i in vars_to_host and j in facs_to_host:
                pb += alphas[((i, j), k)] <= xs[(i, k)], "lin1 {}{}{}".format(i, j, k)
                pb += alphas[((i, j), k)] <= fs[(j, k)], "lin2 {}{}{}".format(i, j, k)
                pb += (
                    alphas[((i, j), k)] >= xs[(i, k)] + fs[(j, k)] - 1,
                    "lin3 {}{}{}".format(i, j, k),
                )

            elif i in vars_to_host and j not in facs_to_host:
                # Var is free, factor is already hosted
                if already_assigned.agent_for(j) == k:
                    pb += alphas[((i, j), k)] == xs[(i, k)]
                else:
                    pb += alphas[((i, j), k)] == 0

            elif i not in vars_to_host and j in facs_to_host:
                # if i is not in vars_vars_to_host, it means that it's a
                # computation that is already hosted (from  hints)
                if already_assigned.agent_for(i) == k:
                    pb += alphas[((i, j), k)] == fs[(j, k)]
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

    # the timeout for the solver must be minored by the time spent to build the pb:
    remaining_time = round(timeout - (time.time() - start_t)) - 2

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

            agt_rels = [j for j, ka in fs if ka == k and pulp.value(fs[(j, ka)]) == 1]
            comp_dist.host_on_agent(k, agt_rels)
        return comp_dist


def _build_alphaijk_binvars(cg: ComputationsFactorGraph, agents_names: Iterable[str]):
    # As these variables are only used in the objective function,
    # when optimizing communication cost, we only need them when (i,j) is an
    # edge in the factor graph
    alphas = LpVariable.dict(
        "a",
        ([(link.variable_node, link.factor_node) for link in cg.links], agents_names),
        cat=LpBinary,
    )
    return alphas


def _build_fs_binvar(facs_to_host, agents_names):
    if not facs_to_host:
        return {}
    return LpVariable.dict("f", (facs_to_host, agents_names), cat=LpBinary)


def _build_xs_binvar(vars_to_host, agents_names):
    if not vars_to_host:
        return {}
    return LpVariable.dict("x", (vars_to_host, agents_names), cat=LpBinary)


def secp_dist_objective_function(
    cg: ComputationsFactorGraph, communication_load, alphas, agents_names
):
    # The objective function is the negated sum of the communication cost on
    # the links in the factor graph.
    return lpSum(
        [
            -communication_load(cg.computation(link.variable_node), link.factor_node)
            * alphas[((link.variable_node, link.factor_node), k)]
            for link in cg.links
            for k in agents_names
        ]
    )


def secp_computation_memory_in_cg(
    computation_name: str, cg: ComputationsFactorGraph, computation_memory
):
    computation = cg.computation(computation_name)
    l = computation_memory(computation)
    return l
