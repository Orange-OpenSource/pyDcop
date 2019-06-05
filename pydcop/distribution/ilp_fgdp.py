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


import logging
from collections import defaultdict
from itertools import combinations
from typing import List, Iterable, Dict, Callable

from pulp import LpMinimize, LpVariable, LpProblem, LpBinary, lpSum, \
    GLPK_CMD, value, LpStatusOptimal

from pydcop.computations_graph.factor_graph import ComputationsFactorGraph
from pydcop.dcop.objects import AgentDef

logger = logging.getLogger('distribution.ilpfgdp')

from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.distribution.objects import Distribution, DistributionHints, \
    ImpossibleDistributionException


"""
IL-FGDP distribution

FGDP : Factor Graph distribution problem
=> only work with factor graphs !


This distribution model only takes communication cost into account, and only through 
message size (i.e. there is no specific cost associated with a given route).  


Designed for an article for OPTMAS 2017 (best paper) 

based on ILP, optimize for communication wrt capacity.

"""


def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory=None,
               communication_load=None):
    """
    Generate a distribution for the dcop.

    :param computation_graph: a ComputationGraph
    :param agentsdef: the agents definitions
    :param hints: a DistributionHints
    :param computation_memory: a function that takes a computation node as an 
      argument and return the memory footprint for this
    :param link_communication: a function that takes a Link as an argument 
      and return the communication cost of this edge
    """
    if computation_memory is None or communication_load is None:
        raise ImpossibleDistributionException('LinearProg distribution requires '
                         'computation_memory and link_communication functions')

    agents = list(agentsdef)

    # In order to remove (latter on) distribution hints, we interpret
    # hosting costs of 0 as a "must host" relationship
    must_host = defaultdict(lambda : [])
    for agent in agentsdef:
        for comp in computation_graph.node_names():
            if agent.hosting_cost(comp) == 0:
                must_host[agent.name].append(comp)
    logger.debug(f"Must host: {must_host}")

    return factor_graph_lp_model(computation_graph, agents, must_host,
                                 computation_memory, communication_load)


def distribution_cost(distribution: Distribution,
                      computation_graph: ComputationGraph,
                      agentsdef: Iterable[AgentDef],
                      computation_memory: Callable[[ComputationNode], float],
                      communication_load: Callable[[ComputationNode, str],
                                                   float]) -> float:
    """
    Compute the cost for a distribution.

    In this model, the cost only includes the communication costs based on message size.

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
    # No hosting and route cost here, as this distribution only takes message size
    # into account:
    # route = route_fonc(agentsdef)
    # hosting_cost = hosting_cost_func(agentsdef)


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

def distribute_remove(secp, current_distribution, removed_device):
    # TODO à implémenter !
    # FIXME : only take neighbors agents as variable ?
    raise NotImplementedError()


def distribute_add(secp, new_device, current_distribution,
                   connected_models=None):
    # TODO à implémenter !
    # FIXME : only take neighbors agents as variable ?
    raise NotImplementedError()


def factor_graph_lp_model(cg: ComputationsFactorGraph,
                          agents: List[AgentDef],
                          must_host: Dict[str, List],
                          computation_memory=None,
                          communication_load=None):
    """
    To distribute we need:
    * com : the communication cost of an edge between a var and a fact
    * mem_var : the memory footprint of a variable computation
    * mem_fac : the memory footprint of a factor computation
    
    These function depends on the algorithm.

    Here    
    * mem_var and mem_fac are given by the computation_memory method.
    * com is given by computation_memory

    :return:
    """
    variables = [n for n in cg.nodes if n.type == 'VariableComputation']
    factors = [n for n in cg.nodes if n.type == 'FactorComputation']

    agents = list(agents)
    agents_names = [a.name for a in agents]

    fixed_dist = Distribution(must_host)

    # Only keep computations for which we actually need to find an agent.
    vars_to_host = [v.name for v in variables
                    if not fixed_dist.has_computation(v.name)]
    facs_to_host = [f.name for f in factors
                    if not fixed_dist.has_computation(f.name)]

    # x_i^k : binary variable indicating if var x_i is hosted on agent a_k.
    xs = _build_xs_binvar(vars_to_host, agents_names)
    # f_j^k : binary variable indicating if factor f_j is hosted on agent a_k.
    fs = _build_fs_binvar(facs_to_host, agents_names)
    # alpha_ijk : binary variable indicating if  x_i and f_j are both on a_k.
    alphas = _build_alphaijk_binvars(cg, agents_names)

    # LP problem with objective function (total communication cost).
    pb = LpProblem('distribution', LpMinimize)
    pb += _objective_function(cg, communication_load, alphas,
                              agents_names), 'Communication costs'
    # Constraints.
    # All variable computations must be hosted:
    for i in vars_to_host:
        pb += lpSum([xs[(i, k)] for k in agents_names]) == 1, \
              'var {} is hosted'.format(i)

    # All factor computations must be hosted:
    for j in facs_to_host:
        pb += lpSum([fs[(j, k)] for k in agents_names]) == 1, \
              'factor {} is hosted'.format(j)

    # Each agent must host at least one computation:
    # We only need this constraints for agents that do not already host a
    # computation:
    empty_agents = [a for a in agents_names if not must_host[a]]
    for k in empty_agents:
        pb += lpSum([xs[(i, k)] for i in vars_to_host]) + \
              lpSum([fs[(j, k)] for j in facs_to_host]) >= 1, \
              'atleastone {}'.format(k)

    # Memory capacity constraint for agents
    for a in agents:
        # Decrease capacity for already hosted computations
        capacity = a.capacity - \
                   sum([_computation_memory_in_cg(c, cg, computation_memory)
                        for c in must_host[a.name]])

        pb += lpSum([_computation_memory_in_cg(i, cg, computation_memory) *
                     xs[(i, a.name)] for i in vars_to_host]) \
            + lpSum([_computation_memory_in_cg(j, cg, computation_memory) *
                     fs[(j, a.name)] for j in facs_to_host]) <= capacity, \
            'memory {}'.format(a.name)

    # Linearization constraints for alpha_ijk.
    for link in cg.links:
        i, j = link.variable_node, link.factor_node
        for k in agents_names:

            if i in vars_to_host and j in facs_to_host:
                pb += alphas[((i, j), k)] <= xs[(i, k)], \
                    'lin1 {}{}{}'.format(i, j, k)
                pb += alphas[((i, j), k)] <= fs[(j, k)], \
                    'lin2 {}{}{}'.format(i, j, k)
                pb += alphas[((i, j), k)] >= xs[(i, k)] + fs[(j, k)] - 1, \
                    'lin3 {}{}{}'.format(i, j, k)

            elif i in vars_to_host and j not in facs_to_host:
                # Var is free, factor is already hosted
                if fixed_dist.agent_for(j) == k:
                    pb += alphas[((i, j), k)] == xs[(i, k)]
                else:
                    pb += alphas[((i, j), k)] == 0

            elif i not in vars_to_host and j in facs_to_host:
                # if i is not in vars_vars_to_host, it means that it's a
                # computation that is already hosted (from  hints)
                if fixed_dist.agent_for(i) == k:
                    pb += alphas[((i, j), k)] == fs[(j, k)]
                else:
                    pb += alphas[((i, j), k)] == 0

            else:
                # i and j are both alredy hosted
                if fixed_dist.agent_for(i) == k and fixed_dist.agent_for(j) \
                        == k:
                    pb += alphas[((i, j), k)] == 1
                else:
                    pb += alphas[((i, j), k)] == 0

    # Now solve our LP
    # status = pb.solve(GLPK_CMD())
    # status = pb.solve(GLPK_CMD(mip=1))
    # status = pb.solve(GLPK_CMD(mip=0, keepFiles=1,
    #                                options=['--simplex', '--interior']))
    status = pb.solve(GLPK_CMD(keepFiles=0, msg=False, options=['--pcost']))

    if status != LpStatusOptimal:
        raise ImpossibleDistributionException("No possible optimal"
                                              " distribution ")
    else:
        logger.debug('GLPK cost : %s', value(pb.objective))

        comp_dist = fixed_dist
        for k in agents_names:

            agt_vars = [i for i, ka in xs
                        if ka == k and value(xs[(i, ka)]) == 1]
            comp_dist.host_on_agent(k, agt_vars)

            agt_rels = [j for j, ka in fs
                        if ka == k and value(fs[(j, ka)]) == 1]
            comp_dist.host_on_agent(k, agt_rels)
        return comp_dist


def _build_alphaijk_binvars(cg: ComputationsFactorGraph, agents_names: Iterable[
    str]):
    # As these variables are only used in the objective function,
    # when optimizing communication cost, we only need them when (i,j) is an
    # edge in the factor graph
    alphas = LpVariable.dict('a',
                             ([(link.variable_node,
                                link.factor_node) for
                               link in cg.links],
                              agents_names),
                             cat=LpBinary)
    return alphas


def _build_fs_binvar(facs_to_host, agents_names):
    if not facs_to_host:
        return {}
    return LpVariable.dict('f', (facs_to_host, agents_names), cat=LpBinary)


def _build_xs_binvar(vars_to_host, agents_names):
    if not vars_to_host:
        return {}
    return LpVariable.dict('x', (vars_to_host, agents_names), cat=LpBinary)


def _objective_function(cg: ComputationGraph, communication_load,
                        alphas, agents_names):
    # The objective function is the negated sum of the communication cost on
    # the links in the factor graph.
    return lpSum([-communication_load(cg.computation(link.variable_node),
                                      link.factor_node ) *
                  alphas[((link.variable_node, link.factor_node), k)]
                  for link in cg.links for k in agents_names])


def _computation_memory_in_cg(computation_name: str,
                              cg: ComputationGraph, computation_memory):
    computation = cg.computation(computation_name)
    l = computation_memory(computation)
    return l
