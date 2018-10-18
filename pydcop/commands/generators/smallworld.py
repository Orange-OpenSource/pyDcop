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


import logging, os
from importlib import import_module

import networkx as nx

from pydcop.algorithms import load_algorithm_module
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable, Domain, AgentDef
from pydcop.dcop.relations import (
    NAryMatrixRelation,
    assignment_matrix,
    random_assignment_matrix,
)
from pydcop.dcop.yamldcop import dcop_yaml

logger = logging.getLogger("pydcop.generate")


def generate_small_world(args):
    logger.debug("generate small world problem %s ", args)

    # Erdős-Rényi graph aka binomial graph.
    graph = nx.barabasi_albert_graph(args.num, 2)

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # nx.draw(graph)  # default spring_layout
    # plt.show()

    domain = Domain("d", "d", range(args.domain))
    variables = {}
    agents = {}
    for n in graph.nodes:
        v = Variable(var_name(n), domain)
        variables[v.name] = v
        logger.debug("Create var for node %s : %s", n, v)

    constraints = {}
    for i, (n1, n2) in enumerate(graph.edges):
        v1 = variables[var_name(n1)]
        v2 = variables[var_name(n2)]
        values = random_assignment_matrix([v1, v2], range(args.range))
        c = NAryMatrixRelation([v1, v2], values, name=c_name(n1, n2))
        logger.debug("Create constraints for edge (%s, %s) : %s", v1, v2, c)
        constraints[c.name] = c

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

    footprints = {n.name: algo_module.computation_memory(n) for n in cg.nodes}
    f_vals = footprints.values()
    logger.info(
        "%s computations, footprint: \n  sum: %s, avg: %s max: %s, " "min: %s",
        len(footprints),
        sum(f_vals),
        sum(f_vals) / len(footprints),
        max(f_vals),
        min(f_vals),
    )

    default_hosting_cost = 2000
    small_agents = [agt_name(i) for i in range(75)]
    small_capa, avg_capa, big_capa = 40, 200, 1000
    avg_agents = [agt_name(i) for i in range(75, 95)]
    big_agents = [agt_name(i) for i in range(95, 100)]
    hosting_factor = 10

    for a in small_agents:
        # communication costs with all other agents
        comm_costs = {other: 6 for other in small_agents if other != a}
        comm_costs.update({other: 8 for other in avg_agents})
        comm_costs.update({other: 10 for other in big_agents})
        # hosting cost for all computations
        hosting_costs = {}
        for n in cg.nodes:
            # hosting_costs[n.name] = hosting_factor * \
            #                         abs(small_capa -footprints[n.name])
            hosting_costs[n.name] = footprints[n.name] / small_capa

        agt = AgentDef(
            a,
            default_hosting_cost=default_hosting_cost,
            hosting_costs=hosting_costs,
            default_route=10,
            routes=comm_costs,
            capacity=small_capa,
        )
        agents[agt.name] = agt
        logger.debug("Create small agt : %s", agt)

    for a in avg_agents:
        # communication costs with all other agents
        comm_costs = {other: 8 for other in small_agents}
        comm_costs.update({other: 2 for other in avg_agents if other != a})
        comm_costs.update({other: 4 for other in big_agents})
        # hosting cost for all computations
        hosting_costs = {}
        for n in cg.nodes:
            # hosting_costs[n.name] = hosting_factor * \
            #                         abs(avg_capa - footprints[n.name])
            hosting_costs[n.name] = footprints[n.name] / avg_capa

        agt = AgentDef(
            a,
            default_hosting_cost=default_hosting_cost,
            hosting_costs=hosting_costs,
            default_route=10,
            routes=comm_costs,
            capacity=avg_capa,
        )
        agents[agt.name] = agt
        logger.debug("Create avg agt : %s", agt)

    for a in big_agents:
        # communication costs with all other agents
        comm_costs = {other: 10 for other in small_agents}
        comm_costs.update({other: 4 for other in avg_agents})
        comm_costs.update({other: 1 for other in big_agents if other != a})
        # hosting cost for all computations
        hosting_costs = {}
        for n in cg.nodes:
            hosting_costs[n.name] = footprints[n.name] / big_capa

        agt = AgentDef(
            a,
            default_hosting_cost=default_hosting_cost,
            hosting_costs=hosting_costs,
            default_route=10,
            routes=comm_costs,
            capacity=big_capa,
        )
        agents[agt.name] = agt
        logger.debug("Create big agt : %s", agt)

    dcop = DCOP(
        "graph coloring",
        "min",
        domains={"d": domain},
        variables=variables,
        agents=agents,
        constraints=constraints,
    )

    if args.output:
        outputfile = args.output[0]
        write_in_file(outputfile, dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


def agt_name(i: int):
    return "a{:02d}".format(i)


def var_name(i: int):
    return "v{:03d}".format(i)


def c_name(i: int, j: int):
    return "c{:03d}_{:03d}".format(i, j)


def write_in_file(filename: str, dcop_str: str):
    path = "/".join(filename.split("/")[:-1])

    if (path != "") and (not os.path.exists(path)):
        os.makedirs(path)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(dcop_str)
