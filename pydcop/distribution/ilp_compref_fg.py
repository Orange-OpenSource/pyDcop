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

Distribution optimizing for communication and agents hosting costs

for AAMAS 2018

We use an interger linear program to find and optimal distribution of
computation over the agents.
The objective is a weighted sum of the communication costs and the hosting
costs.

Communication cost depends on the message size between two computations in
the computation graph and on a route cost factor that characterize the cost of
sending a message over a route between two agents.

Note: this distribution methods honors the agent's capacity constraints but
does no use the distribution hints (if some are given, they are just ignored).

"""


import logging
from typing import Callable, Iterable
from itertools import combinations

from pulp.constants import LpBinary, LpMinimize, LpStatusOptimal
from pulp.pulp import LpVariable, LpProblem, lpSum, value, \
    LpAffineExpression
from pulp.solvers import GLPK_CMD

from pydcop.computations_graph.objects import ComputationGraph, Link, \
    ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import DistributionHints, \
    ImpossibleDistributionException, Distribution

logger = logging.getLogger('distribution.ilp_compref')


# Weight factors when aggregating communication costs and hosting costs in the
# objective function.
# the global objective is built as Comm_cost * RATIO + Hosting_cost * (1-RATIO)
RATIO_HOST_COMM = 0.8


def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory=None,
               communication_load=None) -> Distribution:
    """
    Generate a distribution for the given computation graph.


    :param computation_graph: a ComputationGraph
    :param agentsdef: agents' definitions
    :param hints: a DistributionHints
    :param computation_memory: a function that takes a computation node and its
    Link node as  arguments and return the memory footprint for this node
    :param communication_load: a function that takes a Link as an argument
      and return the communication cost of this edge
    """

    footprint = footprint_fonc(computation_graph, computation_memory)
    capacity = capacity_fonc(agentsdef)
    route = route_fonc(agentsdef)
    msg_load = msg_load_func(computation_graph, communication_load)
    hosting_cost = hosting_cost_func(agentsdef)

    mapping = lp_model(computation_graph, agentsdef, footprint, capacity, route,
                       msg_load, hosting_cost)
    dist = Distribution(mapping)

    return dist


def distribution_cost(distribution: Distribution,
                      computation_graph: ComputationGraph,
                      agentsdef: Iterable[AgentDef],
                      computation_memory: Callable[[ComputationNode], float],
                      communication_load: Callable[[ComputationNode, str],
                                                   float]) -> float:

    route = route_fonc(agentsdef)
    msg_load = msg_load_func(computation_graph, communication_load)
    hosting_cost = hosting_cost_func(agentsdef)

    comm = 0
    agt_names = [a.name for a in agentsdef]
    for l in computation_graph.links:
        # As we support hypergraph, we may have more than 2 ends to a link
        for c1, c2 in combinations(l.nodes, 2):
            a1 = distribution.agent_for(c1)
            a2 = distribution.agent_for(c2)
            comm += route(a1, a2) * msg_load(c1, c2)

    hosting = 0
    for computation in computation_graph.nodes:
        agent = distribution.agent_for(computation.name)
        hosting += hosting_cost(agent, computation.name)

    cost = RATIO_HOST_COMM * comm + (1-RATIO_HOST_COMM) * hosting
    return cost, comm, hosting


def lp_model(cg: ComputationGraph,
             agentsdef: Iterable[AgentDef],
             footprint: Callable[[str], float],
             capacity: Callable[[str], float],
             route: Callable[[str, str], float],
             msg_load: Callable[[str, str], float],
             hosting_cost: Callable[[str, str], float]):

    comp_names = [n.name for n in cg.nodes]

    agt_names = [a.name for a in agentsdef]
    pb = LpProblem('ilp_compref', LpMinimize)

    # One binary variable xij for each (variable, agent) couple
    xs = LpVariable.dict('x', (comp_names, agt_names), cat=LpBinary)

    # One binary variable for computations c1 and c2, and agent a1 and a2
    betas = {}
    count = 0
    for a1, a2 in combinations(agt_names, 2):
        # Only create variables for couple c1, c2 if there is an edge in the
        # graph between these two computations.
        for l in cg.links:
            # As we support hypergraph, we may have more than 2 ends to a link
            for c1, c2 in combinations(l.nodes, 2):
                count += 2
                b = LpVariable('b_{}_{}_{}_{}'.format(c1, a1, c2, a2),
                               cat=LpBinary)
                betas[(c1, a1, c2, a2)] = b
                pb += b <= xs[(c1, a1)]
                pb += b <= xs[(c2, a2)]
                pb += b >= xs[(c2, a2)] + xs[(c1, a1)] - 1

                b = LpVariable('b_{}_{}_{}_{}'.format(c1, a2, c2, a1),
                               cat=LpBinary)
                betas[(c1, a2, c2, a1)] = b
                pb += b <= xs[(c2, a1)]
                pb += b <= xs[(c1, a2)]
                pb += b >= xs[(c1, a2)] + xs[(c2, a1)] - 1

    # Set objective: communication + hosting_cost
    pb += _objective(xs, betas, route, msg_load, hosting_cost), \
        'Communication costs and prefs'

    # Adding constraints:
    # Constraints: Memory capacity for all agents.
    for a in agt_names:
        pb += lpSum([footprint(i) * xs[i, a] for i in comp_names])\
              <= capacity(a), \
              'Agent {} capacity'.format(a)

    # Constraints: all computations must be hosted.
    for c in comp_names:
        pb += lpSum([xs[c, a] for a in agt_names]) == 1, \
            'Computation {} hosted'.format(c)

    # solve using GLPK
    status = pb.solve(solver=GLPK_CMD(keepFiles=1, msg=False,
                                      options=['--pcost']))

    if status != LpStatusOptimal:
        raise ImpossibleDistributionException("No possible optimal"
                                              " distribution ")
    logger.debug('GLPK cost : %s', value(pb.objective))

    # print('BETAS:')
    # for c1, a1, c2, a2 in betas:
    #     print('  ', c1, a1, c2, a2, value(betas[(c1, a1, c2, a2)]))
    #
    # print('XS:')
    # for c, a in xs:
    #     print('  ', c, a, value(xs[(c, a)]))

    mapping = {}
    for k in agt_names:
        agt_computations = [i for i, ka in xs
                            if ka == k and value(xs[(i, ka)]) == 1]
        # print(k, ' -> ', agt_computations)
        mapping[k] = agt_computations
    return mapping


def msg_load_func(cg: ComputationGraph,
                  communication_load: Callable[[ComputationNode, str], float])\
        -> Callable[[str, str], float]:
    def msg_load(c1: str, c2: str) -> float:
        load = 0
        links = cg.links_for_node(c1)
        for l in links:
            if c2 in l.nodes:
                load += communication_load(cg.computation(c1), c2)
        return load
    return msg_load


def capacity_fonc(agents_def: Iterable[AgentDef])\
        -> Callable[[str], float]:
    """
    :param agents_def:
    :return: a function that gives the agent's capacity given it's name
    """
    def capacity(agent_name):
        for a in agents_def:
            if a.name == agent_name:
                return a.capacity
    return capacity


def route_fonc(agents_def: Iterable[AgentDef])\
        -> Callable[[str], float]:

    def route(a1_name: str, a2_name: str):
        for a in agents_def:
            if a.name == a1_name:
                return a.route(a2_name)
    return route


def footprint_fonc(cg: ComputationGraph,
                   computation_memory: Callable[[ComputationNode,
                                                 Iterable[Link]],
                                                float])\
        -> Callable[[str], float]:
    """
    :param cg: the computation graph
    :param computation_memory: a function giving a memory footprint from a
    computation node and a set of link in the computation graph
    :return: a function that returns the memory footprint of a computation
    given it's name
    """
    def footprint(computation_name: str):
        c = cg.computation(computation_name)
        return computation_memory(c)
    return footprint


def hosting_cost_func(agts_def: Iterable[AgentDef])\
        -> Callable[[str, str], float]:
    """

    :param agts_def: the AgentsDef
    :return: a function that returns the hosting cost for agt, comp
    """
    def cost(agt_name: str, comp_name: str):
        for a in agts_def:
            if a.name == agt_name:
                return a.hosting_cost(comp_name)
    return cost


def _objective(xs, betas, route, msg_load, hosting_cost):
    # We want to minimize communication and hosting costs
    # Objective is the communication + hosting costs
    comm = LpAffineExpression()
    for c1, a1, c2, a2 in betas:
        comm += route(a1, a2) * msg_load(c1, c2) * betas[(c1, a1, c2, a2)]

    costs = lpSum([hosting_cost(a, c) * xs[(c, a)] for c, a in xs])

    return lpSum([RATIO_HOST_COMM * comm ,  (1-RATIO_HOST_COMM) * costs])
