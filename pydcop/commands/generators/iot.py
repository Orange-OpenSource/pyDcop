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
This generator generates both a dcop and its initial distribution.

"""


import logging, os
from importlib import import_module
from random import randint
from typing import List, Tuple, Dict, Callable

import networkx as nx
from collections import defaultdict

import yaml
from pulp.constants import LpBinary, LpMinimize, LpStatusOptimal
from pulp.pulp import LpVariable, LpProblem, lpSum, value, LpAffineExpression
from pulp.solvers import GLPK_CMD

from pydcop.algorithms import load_algorithm_module
from pydcop.computations_graph.factor_graph import (
    VariableComputationNode,
    FactorComputationNode,
)
from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable, Domain, AgentDef
from pydcop.dcop.relations import (
    NAryMatrixRelation,
    assignment_matrix,
    random_assignment_matrix,
    Constraint,
)
from pydcop.dcop.yamldcop import dcop_yaml
from pydcop.distribution import ilp_compref
from pydcop.distribution.objects import Distribution, ImpossibleDistributionException

logger = logging.getLogger("pydcop.generate")

RATIO_HOST_COMM = 0.8


def generate_iot(args):
    print("generate iot ", args.output)

    # Constraints and variables with a power-law constraint graph:
    variables, constraints, domain = generate_powerlaw_var_constraints(
        args.num, args.domain, args.range
    )

    # Build a dcop and computation graph with no agents, just to be able to
    # compute the footprint of computations:
    dcop = DCOP(
        "graph coloring",
        "min",
        domains={"d": domain},
        variables=variables,
        agents={},
        constraints=constraints,
    )
    graph_module = import_module("pydcop.computations_graph.factor_graph")
    cg = graph_module.build_computation_graph(dcop)
    algo_module = load_algorithm_module("maxsum")

    footprints = {c.name: algo_module.computation_memory(c) for c in cg.nodes}

    # Generate an agent for each variable computation and assign the
    # computation to that agent.
    agents = {}  # type: Dict[str, AgentDef]
    mapping = defaultdict(lambda: [])  # type: Dict[str, List[str]]
    for comp in cg.nodes:
        if isinstance(comp, VariableComputationNode):
            a_name = agt_name(comp.name)
            agt = AgentDef(
                a_name,
                capacity=footprints[comp.name] * 100,
                default_hosting_cost=10,
                hosting_costs=agt_hosting_costs(comp, cg),
                default_route=1,
                routes=agt_route_costs(comp, cg),
            )
            logger.debug(
                "Create agent %s for computation %s with capacity %s",
                agt.name,
                comp.name,
                agt.capacity,
            )
            agents[agt.name] = agt
            mapping[agt.name].append(comp.name)

    # Now, we have created all the agents and distributed all the variables
    # let's distribute the factor computations.
    msg_load = msg_load_func(cg, algo_module.communication_load)
    factor_mapping = distribute_factors(agents, cg, footprints, mapping, msg_load)
    for a in mapping:
        mapping[a].extend(factor_mapping[a])

    dcop = DCOP(
        "graph coloring",
        "min",
        domains={"d": domain},
        variables=variables,
        agents=agents,
        constraints=constraints,
    )

    distribution = Distribution(mapping)

    if args.output:
        outputfile = args.output
        write_in_file(outputfile, dcop_yaml(dcop))

        dist = distribution.mapping()
        cost = ilp_compref.distribution_cost(
            distribution,
            cg,
            dcop.agents.values(),
            computation_memory=algo_module.computation_memory,
            communication_load=algo_module.communication_load,
        )

        result = {
            "inputs": {
                "dist_algo": "io_problem",
                "dcop": args.output,
                "graph": "factor_graph",
                "algo": "maxsum",
            },
            "distribution": dist,
            "cost": cost,
        }
        outputfile = "dist_" + args.output
        write_in_file(outputfile, yaml.dump(result))
    else:
        print(dcop_yaml(dcop))


def generate_powerlaw_var_constraints(
    num_var: int, domain_size: int, constraint_range: int
) -> Tuple[Dict[str, Variable], Dict[str, Constraint], Domain]:
    """
    Generate variables and constraints for a power-law based constraints
    graph.

    All constraints are binary and the graph is generated using the Barabasi
    Albert method.

    Parameters
    ----------
    num_var: int
        number of variables
    domain_size:  int
        size of the domain of the variables
    constraint_range: int
        range in which constraints take their value (uniform random value of
        ech possible assignment).

    Returns
    -------
    A tuple with variables, constraints and domain.
    """

    # Use a barabasi powerlaw based constraints graph
    graph = nx.barabasi_albert_graph(num_var, 2)

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # nx.draw(graph)  # default spring_layout
    # plt.show()

    domain = Domain("d", "d", range(domain_size))
    variables = {}
    for n in graph.nodes:
        v = Variable(var_name(n), domain)
        variables[v.name] = v
        logger.debug("Create var for node %s : %s", n, v)

    constraints = {}
    for i, (n1, n2) in enumerate(graph.edges):
        v1 = variables[var_name(n1)]
        v2 = variables[var_name(n2)]
        values = random_assignment_matrix([v1, v2], range(constraint_range))
        c = NAryMatrixRelation([v1, v2], values, name=c_name(n1, n2))
        logger.debug("Create constraints for edge (%s, %s) : %s", v1, v2, c)
        constraints[c.name] = c

    logger.info(
        "Generates %s variables and %s constraints in a powerlaw" "network",
        len(variables),
        len(constraints),
    )

    return variables, constraints, domain


def agt_route_costs(
    var_comp: ComputationNode, cg: ComputationGraph
) -> Dict[str, float]:
    """
    Generate route cost between the agent hosting var_comp and all other agents.

    Parameters
    ----------
    var_comp: ComputationNode
        the Variable computation hosted by the agent
    cg: computation graph

    Returns
    -------
    a dict containing the route cost to all other agents
    """
    routes = {}
    degree_v = len(var_comp.neighbors)
    for neighbor in var_comp.neighbors:
        # var_com is a variable computation, each neighbor will be a factor
        # which has two neighbor, var_comp and another variable
        neigh_var = next(
            c for c in cg.computation(neighbor).neighbors if c != var_comp.name
        )
        degree_n = len(cg.computation(neigh_var).neighbors)
        route = (1 + abs(degree_n - degree_v)) / (degree_n + degree_v)
        routes[agt_name(neigh_var)] = route
    return routes


def agt_hosting_costs(
    var_comp: ComputationNode, cg: ComputationGraph
) -> Dict[str, float]:
    """
    Build the hosting costs dict for the agent hosting the variable whose
    computation is var_comp.

    Hosting costs are uniform random for all computations, except for
    var_comp for which it is zero.

    Parameters
    ----------
    var_comp: ComputationNode
        Computation for the variable hosted by the agent
    cg:
        computation graph

    Returns
    -------
    a dict {computation_name: float} representing the hosting cost for each
    computation on this agent.
    """
    hosting_costs = {c.name: randint(0, 10) for c in cg.nodes}
    hosting_costs[var_comp.name] = 0
    return hosting_costs


def distribute_factors(
    agents: Dict[str, AgentDef],
    cg: ComputationGraph,
    footprints: Dict[str, float],
    mapping: Dict[str, List[str]],
    msg_load: Callable[[str, str], float],
) -> Dict[str, List[str]]:
    """
    Optimal distribution of factors on agents.

    Parameters
    ----------
    cg: computations graph

    agents: dict
        a dict {agent_name : AgentDef} containing all available agents

    Returns
    -------
    a dict { agent_name: list of factor names}
    """
    pb = LpProblem("ilp_factors", LpMinimize)

    # build the inverse mapping var -> agt
    inverse_mapping = {}  # type: Dict[str, str]
    for a in mapping:
        inverse_mapping[mapping[a][0]] = a

    # One binary variable xij for each (variable, agent) couple
    factor_names = [n.name for n in cg.nodes if isinstance(n, FactorComputationNode)]
    xs = LpVariable.dict("x", (factor_names, agents), cat=LpBinary)
    logger.debug("Binary variables for factor distribution : %s", xs)

    # Hard constraints: respect agent's capacity
    for a in agents:
        # Footprint of the variable this agent is already hosting:
        v_footprint = footprints[mapping[a][0]]
        pb += (
            lpSum([footprints[fn] * xs[fn, a] for fn in factor_names])
            <= (agents[a].capacity - v_footprint),
            "Agent {} capacity".format(a),
        )

    # Hard constraints: all computations must be hosted.
    for c in factor_names:
        pb += lpSum([xs[c, a] for a in agents]) == 1, "Factor {} hosted".format(c)

    # 1st objective : minimize communication costs:
    comm = LpAffineExpression()
    for (fn, an_f) in xs:
        for vn in cg.neighbors(fn):
            an_v = inverse_mapping[vn]  # agt hosting neighbor var vn
            comm += agents[an_f].route(an_v) * msg_load(vn, fn) * xs[(fn, an_f)]

    # 2st objective : minimize hosting costs
    hosting = lpSum([agents[a].hosting_cost(c) * xs[(c, a)] for c, a in xs])

    # agregate the two objectives using RATIO_HOST_COMM
    pb += lpSum([RATIO_HOST_COMM * comm, (1 - RATIO_HOST_COMM) * hosting])

    # solve using GLPK and convert to mapping { agt_name : [factors names]}
    status = pb.solve(solver=GLPK_CMD(keepFiles=1, msg=False, options=["--pcost"]))
    if status != LpStatusOptimal:
        raise ImpossibleDistributionException(
            "No possible optimal distribution for factors"
        )
    logger.debug("GLPK cost : %s", value(pb.objective))
    mapping = {}  # type: Dict[str, List[str]]
    for k in agents:
        agt_computations = [i for i, ka in xs if ka == k and value(xs[(i, ka)]) == 1]
        # print(k, ' -> ', agt_computations)
        mapping[k] = agt_computations
    logger.debug("Factors distribution : %s ", mapping)
    return mapping


def agt_name(var_name: str):
    return "a{}".format(var_name[1:])


def var_name(i: int):
    return "v{:03d}".format(i)


def c_name(i: int, j: int):
    return "c{:03d}_{:03d}".format(i, j)


def msg_load_func(
    cg: ComputationGraph, communication_load: Callable[[ComputationNode, str], float]
) -> Callable[[str, str], float]:
    def msg_load(c1: str, c2: str) -> float:
        load = 0
        links = cg.links_for_node(c1)
        for l in links:
            if c2 in l.nodes:
                load += communication_load(cg.computation(c1), c2)
        return load

    return msg_load


def write_in_file(filename: str, dcop_str: str):
    path = "/".join(filename.split("/")[:-1])

    if (path != "") and (not os.path.exists(path)):
        os.makedirs(path)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(dcop_str)
