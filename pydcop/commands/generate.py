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
.. _pydcop_commands_generate:

pydcop generate
===============

Problem generator for benchmarks.

::

    pydcop generate [--output <file>] <problem-type> ...


The ``generate`` command is still a work in progress,
it currently generate problems for the following types:

.. toctree::
   :maxdepth: 1

   generate/graphcoloring
   generate/meetingscheduling
   generate/ising
   generate/secp


Planned
-------

* mixed constraints: generate dcops with both hard and soft constraints
* IoT-like problems
* sensor network




"""
# References:
#     https://manual.frodo-ai.tech/FRODO_User_Manual.html#x1-65000B
#     http://teamcore.usc.edu/dcop/
#
# Graph coloring
#   https://stackoverflow.com/questions/20171901/how-to-generate-random-graphs
#   http://networkx.readthedocs.io/en/networkx-1.11/reference/generators.html
#
# TODO : other type of problems that we could/should implement
#
# * meeting scheduling
# * sensor networks (from http://teamcore.usc.edu/DCOP/ADOPT-AAMAS04-data.tar.gz
#   and http://teamcore.usc.edu/papers/2004/RealWorldDCOP.pdf)


import logging
import random
from math import floor
from collections import defaultdict
from typing import Dict, List
from typing import Tuple

import networkx as nx
import os

# from numpy.random import random
from pydcop.commands.generators import graphcoloring, meetingscheduling, ising, agents, \
    scenario
from pydcop.commands.generators.iot import generate_iot
from pydcop.commands.generators.secp import generate_secp, parser_secp
from pydcop.commands.generators.smallworld import generate_small_world
from pydcop.dcop.objects import VariableDomain, Variable, AgentDef
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import relation_from_str
from pydcop.dcop.yamldcop import dcop_yaml

logger = logging.getLogger("pydcop.cli.generate")


def set_parser(main_subparsers):
    parser = main_subparsers.add_parser("generate", help="Generate Random problems")
    parser.set_defaults(func=run_cmd)
    parser.add_argument(
        "-c",
        "--correct_density",
        action="store_true",
        default=False,
        help='In case you use the pattern "/density=d/" ('
        "where d "
        "is a float) in the path of the output file, "
        "this argument helps you to correct the density "
        "if the generated graph has a different density."
        "However this rounds density to 1 decimal.",
    )
    # parser.add_argument('-o', '--output', nargs=1, default=None,
    #                     help='Determines if  the output is printed in console '
    #                          'or returned in the code. If not used, '
    #                          'the result is printed, else it is stored in the'
    #                          'file given as argument.')

    subparsers = parser.add_subparsers(
        title="Problems",
        dest="problem",
        description="The type of problem you " "want to generate",
    )
    # parser.set_defaults(func=run_cmd)

    graphcoloring.init_cli_parser(subparsers)
    meetingscheduling.init_cli_parser(subparsers)
    ising.init_cli_parser(subparsers)

    agents.init_cli_parser(subparsers)
    scenario.init_cli_parser(subparsers)
    parser_mixed_problem(subparsers)

    parser_ising_soft(subparsers)

    parser_iot_problem(subparsers)

    parser_secp(subparsers)


def parser_iot_problem(subparsers):
    parser = subparsers.add_parser(
        "iot",
        help="generate a DCOP modelling a "
        "typical IoT problem. All constraints "
        "are binary and cost are random.",
    )
    parser.set_defaults(func=generate_iot)
    parser.add_argument(
        "-d",
        "--domain",
        type=int,
        required=True,
        help="domain of the variables domain: 0, 1, ..., d-1",
    )
    parser.add_argument(
        "-n", "--num", type=int, required=True, help="number of variables in the graph"
    )
    # parser.add_argument('-p', '--p', type=float, required=True,
    #                     help='probability of edge creation')
    parser.add_argument(
        "-r", "--range", type=int, default=10, help="range of the constraints values"
    )
    # parser.add_argument('-a', '--agents', type=int, required=True,
    #                     help='number of agents')


def parser_ising_soft(subparsers):
    parser = subparsers.add_parser(
        "ising_soft",
        help="Generates a random problem with soft "
        "constraints and an ising-based "
        "constraints graph",
    )
    parser.set_defaults(func=generate_ising)
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=True,
        help="size of the izing graph (which will contains s*s "
        "variables and 2*s*s constraints)",
    )
    parser.add_argument(
        "-r",
        "--range",
        type=int,
        required=True,
        help="range of the variables domain: 0, 1, ..., r-1",
    )
    parser = subparsers.add_parser(
        "small_world",
        help="generate a DCOP with a small world "
        "constraint graph. All constraints "
        "are binary and cost are random.",
    )
    parser.set_defaults(func=generate_small_world)
    parser.add_argument(
        "-d",
        "--domain",
        type=int,
        required=True,
        help="domain of the variables domain: 0, 1, ..., d-1",
    )
    parser.add_argument(
        "-n", "--num", type=int, required=True, help="number of variables in the graph"
    )
    # parser.add_argument('-p', '--p', type=float, required=True,
    #                     help='probability of edge creation')
    parser.add_argument(
        "-r", "--range", type=int, default=10, help="range of the constraints values"
    )
    # parser.add_argument('-a', '--agents', type=int, required=True,
    #                     help='number of agents')


def parser_mixed_problem(subparsers):
    parser = subparsers.add_parser(
        "mixed_problem", help="generate a DCOP graph coloring " "problem."
    )
    parser.set_defaults(func=generate_mixed_problem)
    parser.add_argument(
        "-v", "--variable_count", type=int, required=True, help="number of variables"
    )
    parser.add_argument(
        "-c",
        "--constraint_count",
        type=int,
        required=True,
        help="number of constraints",
    )
    parser.add_argument(
        "-H",
        "--hard_constraint",
        type=float,
        required=True,
        help="proportion of hard constraints",
    )
    parser.add_argument(
        "-A",
        "--arity",
        type=int,
        required=False,
        default=2,
        help="The maximum arity of the constraints",
    )
    parser.add_argument(
        "-r",
        "--range",
        type=int,
        required=True,
        help="range of the variables domain: 0, 1, ..., r-1",
    )
    parser.add_argument(
        "-d", "--density", type=float, required=True, help="Graph density."
    )
    parser.add_argument(
        "-a",
        "--agents",
        type=int,
        required=False,
        default=None,
        help="number of agents, if not given the number of " "node is used.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        required=False,
        default=0,
        help="Capacity of the agents.",
    )
    # TODO : intensional vs extensive form
    parser.add_argument(
        "-e",
        "--extensive",
        help="generate the problem in extensive form (default "
        "is intentional form) : NOT IMPLEMENTED YET",
    )


def parser_graph_coloring(subparsers):
    parser = subparsers.add_parser(
        "graph_coloring", help="generate a DCOP graph coloring " "problem."
    )
    parser.set_defaults(func=generate_graph_coloring)
    # parser.add_argument('file', type=str, help="file")
    parser.add_argument(
        "-n",
        "--node_count",
        type=int,
        required=True,
        help="number of nodes (variables)",
    )
    parser.add_argument(
        "-c", "--color_count", type=int, required=True, help="number of colors"
    )
    parser.add_argument(
        "-d",
        "--density",
        type=float,
        required=True,
        help="target graph density, due to the way the "
        "graph is generated, the actual graph "
        "density will not be equal to d (especially for "
        "small graphs) but for large enough graph ("
        "or enough graph generation) will tend to it",
    )
    parser.add_argument(
        "-a",
        "--agents",
        type=int,
        required=False,
        default=None,
        help="number of agents, if not given the number of " "node is used.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        required=False,
        default=0,
        help="Capacity of the agents.",
    )
    parser.add_argument(
        "-s",
        "--allow_subgraph",
        action="store_true",
        default=False,
        help="allows (but does not guarantee) the graph to "
        "have several disconnected subgraphs, "
        "by default we generate problem with a single "
        "connected graph",
    )
    # TODO : intentionnal vs extensive form
    parser.add_argument(
        "-e",
        "--extensive",
        action="store_true",
        help="generate the problem in extensive form (default "
        "is intentional form) : NOT IMPLEMENTED YET",
    )
    parser.add_argument(
        "-g",
        "--generator",
        help="NOT IMPLEMENTED YET : graph generation model, "
        "Erdős-Rényi or Watts–Strogatz",
    )


def run_cmd(args):
    print(
        '"pydcop generate" can generate several kind of dcop\n'
        'Run "pydcop generate graphcoloring --help" or \n'
        '    "pydcop generate mixed_problem --help"  \n'
        "for help"
    )


def generate_graph_coloring(args):
    logger.debug("generate_graph_coloring %s ", args)
    node_count = args.node_count
    density = args.density
    color_count = args.color_count
    auto_agents = args.agents is None
    agents_count = node_count if auto_agents else args.agents
    capacity = args.capacity
    logger.info(
        "Generating random graph coloring with %s variables, "
        "%s colors, target density %s and %s agents",
        node_count,
        color_count,
        args.density,
        agents_count,
    )

    # First a random graph
    is_connected = False
    if not args.allow_subgraph:
        while not is_connected:
            graph = nx.gnp_random_graph(args.node_count, density)
            is_connected = nx.is_connected(graph)
    else:
        graph = nx.gnp_random_graph(args.node_count, density)
        is_connected = nx.is_connected(graph)

    real_density = nx.density(graph)
    logger.info(nx.info(graph))
    logger.info("Connected : %s", nx.is_connected(graph))
    logger.info("Density %s", nx.density(graph))

    # Now create a DCOP from the graph
    d = VariableDomain("colors", "color", range(color_count))
    variables = {}
    agents = {}
    for i, node in enumerate(graph.nodes_iter()):
        logger.debug("node %s - %s", node, i)
        name = "v" + str(i)
        variables[name] = Variable(name, d)
        if auto_agents:
            a_name = "a" + str(i)
            agents[a_name] = AgentDef(a_name, capacity)

    if not auto_agents:
        for i in range(agents_count):
            a_name = "a" + str(i)
            agents[a_name] = AgentDef(a_name, capacity)

    constraints = {}
    for i, edge in enumerate(graph.edges_iter()):
        logger.debug("edge %s - %s", edge, i)
        name = "c" + str(i)
        u, v = edge
        expression = "1000 if v{} == v{} else 0".format(u, v)
        constraints[name] = relation_from_str(name, expression, variables.values())
        logger.debug(repr(constraints[name]))

    dcop = DCOP(
        "graph coloring",
        "min",
        domains={"colors": d},
        variables=variables,
        agents=agents,
        constraints=constraints,
    )

    if args.output:
        outputfile = args.output[0]
        if args.correct_density:
            outputfile = correct_density(outputfile, real_density)
        write_in_file(outputfile, dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


# We do not use networkx bipartite generator because, if there are nodes
# without neighbors, the sets of bottom and top nodes are wrongly calculated.
# All the isolated nodes are put in the bottom set. However we need to keep a
#  distinction between constraints and variables here.
# Thus we create random bipartite graphs using a uniform probability to
# choose edges, while decreasing the set of available edges at each picked edge
def generate_mixed_problem(args):
    logger.debug("generate_mixed_problem %s ", args)
    variable_count = args.variable_count
    constraint_count = args.constraint_count
    density = args.density
    real_density = density
    domain_range = args.range
    arity = args.arity
    auto_agents = args.agents is None
    capacity = args.capacity
    agents_count = variable_count if auto_agents else args.agents
    nb_max_edges = constraint_count * min(arity, variable_count)
    edges_count = int(nb_max_edges * density)
    hard_count = int(args.hard_constraint * edges_count)
    logger.info(
        "Generating random DCOP graph with %s variables, whose domain "
        "are [0;%s], %s edges, %s agents, %s hard "
        "constraints and %s soft constraints",
        variable_count,
        domain_range - 1,
        edges_count,
        agents_count,
        hard_count,
        constraint_count - hard_count,
    )

    if arity > variable_count:
        raise ValueError(
            "The arity of a constraint must be at most the "
            "number of variable. Arity: {}, Nb variables: {}".format(
                arity, variable_count
            )
        )

    if hard_count < 0:
        raise ValueError(
            "The argument '-h' (or '--hard_count') must be "
            "between 0 and 1. Currently set to: {}".format(hard_count)
        )
    # Create sets for the bipartite graph
    if constraint_count <= 0:
        raise ValueError(
            "The argument '-c' (or '--constraint_count') must be "
            "strictly positive. Currently set to: {}".format(constraint_count)
        )
    if variable_count < 0:
        raise ValueError(
            "The argument '-v' (or '--variable_count') must be "
            "at least 1. Currently set to: {}".format(variable_count)
        )
    if arity <= 0:
        raise ValueError(
            "The argument '-a' (or '--arity') must be "
            "at least 1. Currently set to: {}".format(arity)
        )

    d = VariableDomain("levels", "level", range(domain_range))
    variables = {}
    agents = {}
    constraints = {}

    if arity == 1:
        if constraint_count != variable_count:
            raise ValueError(
                "For max arity 1 you need the same number of "
                "variables, constraints and edges. You asked "
                "for {} variables and {} constraints.".format(
                    variable_count, constraint_count
                )
            )
        nodes = [i + 1 for i in range(variable_count)]
        constraints_list = [
            ("c{}".format(i + 1), "hard")
            if i < hard_count
            else ("c{}".format(i + 1), "soft")
            for i in range(constraint_count)
        ]
        variables = {}
        constraints = {}
        while len(nodes) != 0:
            n = nodes.pop()
            c = constraints_list.pop(random.randint(0, len(constraints_list) - 1))
            w = choose_weight()
            hard = c[1] == "hard"
            objective = find_objective([w], domain_range - 1, hard)
            if hard:
                expression = (
                    "float('inf') if "
                    + str(w)
                    + "*v"
                    + str(n)
                    + " != "
                    + str(objective)
                    + " else 0"
                )
            else:
                expression = str(w) + "*v" + str(n) + " - " + str(objective)

            v = Variable("v" + str(n), d)
            variables["v" + str(n)] = v
            constraints[c[0]] = relation_from_str(c[0], expression, [v])

            if auto_agents:
                a_name = "a" + str(n)
                agents[a_name] = AgentDef(a_name, capacity)

        if not auto_agents:
            for i in range(agents_count):
                a_name = "a" + str(i)
                agents[a_name] = AgentDef(a_name, capacity)

    elif arity == 2:
        edges_count = int(variable_count * (variable_count - 1) * density / 2)
        if constraint_count != edges_count:
            logger.warning(
                "edges count is different of constraint count ({} "
                "!= {}) but for arity 2, constraints are the deges"
                "of the graph. We use the density ({}) to determine"
                " the number of edges".format(edges_count, constraint_count, density)
            )
        is_connected = False
        while not is_connected:
            graph = nx.gnp_random_graph(variable_count, density)
            is_connected = nx.is_connected(graph)

        # Compute nb of hard constraints regarding the true density
        real_density = nx.density(graph)
        hard_count = (
            args.hard_constraint
            * real_density
            * args.variable_count
            * (args.variable_count + 1)
            / 2
        )

        for i, node in enumerate(graph.nodes_iter()):
            logger.debug("node %s - %s", node, i)
            name = "v" + str(i)
            variables[name] = Variable(name, d)
            if auto_agents:
                a_name = "a" + str(i)
                agents[a_name] = AgentDef(a_name, capacity)

        if not auto_agents:
            for i in range(agents_count):
                a_name = "a" + str(i)
                agents[a_name] = AgentDef(a_name, capacity)

        constraints = {}
        for i, edge in enumerate(graph.edges_iter()):
            logger.debug("edge %s - %s", edge, i)
            name = "c" + str(i)
            u, v = edge
            weights = [round(choose_weight(), 2), round(choose_weight(), 2)]

            # Create hard_constraints
            if i < hard_count:
                objective = round(find_objective(weights, domain_range, True), 2)
                expression = "0 if v{} != v{} else float(" "'inf')".format(u, v)
            else:
                max_val = (weights[0] + weights[1]) * domain_range
                expression = "abs(v{} + v{} - {})".format(
                    u, v, round(random.uniform(0, max_val), 2)
                )

            constraints[name] = relation_from_str(name, expression, variables.values())
            logger.debug(repr(constraints[name]))

    else:
        if edges_count < constraint_count and arity != 1:
            raise ValueError(
                "The number of edges must be greater or equal to the "
                "number of constraints. Otherwise you have unused "
                "constraints. Edges: {}, Constraints: {}".format(
                    edges_count, constraint_count
                )
            )
        nodes = [i for i in list(range(variable_count))]
        constraints = [
            ("c{}".format(i), "hard") if i < hard_count else ("c{}".format(i), "soft")
            for i in list(range(constraint_count))
        ]
        # Randomly add edges
        edges = defaultdict(lambda: [])  # final set of edges

        # Available edges at a given run
        available_edges = {n: constraints.copy() for n in nodes}
        # First, make sure each variable has one constraint
        for n in nodes:
            if constraints:
                node = n
                c = constraints.pop(random.randint(0, len(constraints) - 1))
            else:
                node, c = choose_in_available_edges(available_edges, n)
            add_edge(node, c, available_edges, edges, arity)
            logger.debug("Add edge (%s, %s)", n, c)
        edges_count -= variable_count

        # Second, make sure each constraint is used
        for c in constraints:
            n = random.choice(nodes)
            add_edge(n, c, available_edges, edges, arity)
            edges_count -= 1
            logger.debug("Add edge (%s, %s)", n, c)

        # Third, randomly add remaining constraints
        while edges_count != 0:
            n, c = choose_in_available_edges(available_edges)
            if (n, c) == (None, None):
                # If more edges than possible are asked, returns just the maximum
                # edges (regarding nodes number and constraints arity)
                logger.warning(
                    "%s edges were not added because you asked for too"
                    " many edges regarding the number of constraints ("
                    "%s) and their maximum arity (%s)",
                    edges_count - len(edges),
                    constraint_count,
                    args.arity,
                )
                break
            else:
                add_edge(n, c, available_edges, edges, arity)
                edges_count -= 1

        # Now create a DCOP from the graph
        for i in nodes:
            name = "v" + str(i)
            variables[name] = Variable(name, d)
            if auto_agents:
                a_name = "a" + str(i)
                agents[a_name] = AgentDef(a_name, capacity)

        if not auto_agents:
            for i in range(agents_count):
                a_name = "a" + str(i)
                agents[a_name] = AgentDef(a_name, capacity)

        constraints = {}
        for c, neighbors in edges.items():
            logger.debug("Constraint %s, variables: %s", c, neighbors)
            name, is_hard = c[0], c[1] == "hard"
            c_variables = [variables["v" + str(name)] for name in neighbors]
            addition_string = ""
            first = True
            weights = []
            max_val = 0
            for i in neighbors:
                # Add random weights in constraints
                m = round(choose_weight(), 2)
                weights.append(m)
                max_val += m * (domain_range - 1)
                if not first:
                    addition_string += " + "
                else:
                    first = False
                if m != 1:
                    addition_string += str(m) + "*"
                addition_string += "v" + str(i) + " "

            # To ensure that our hard constraints are achievable, we use
            # the value obtained for a set of random values (in the domain)
            # as the objective.

            objective = round(find_objective(weights, domain_range, is_hard), 2)

            if is_hard:
                expression = (
                    "0 if "
                    + addition_string
                    + " == "
                    + str(objective)
                    + " else float('inf')"
                )
            else:
                const_function = "abs(" + addition_string
                if objective:
                    expression = const_function + " - " + str(objective) + ")"
                else:
                    expression = addition_string

            constraints[name] = relation_from_str(name, expression, c_variables)
            logger.debug(repr(constraints[name]))

    dcop = DCOP(
        "mixed constraints problem",
        "min",
        domains={"levels": d},
        variables=variables,
        constraints=constraints,
        agents=agents,
    )

    if args.output is not None:
        outputfile = args.output[0]
        if args.correct_density:
            outputfile = correct_density(outputfile, real_density)
        write_in_file(outputfile, dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


def choose_in_available_edges(available_edges, n=None):
    """
    Randomly choose an edge among available edges (ie not already existing)
    :param n: A node to be in the edge
    :param available_edges: The dictionary of available_edges for each node {
    node: list of constraints}
    :return: the chosen edge as a couplt (variable node, constraint node)
    """
    if not available_edges:
        return (None, None)
    if n is not None:
        node = n
        c = random.choice(available_edges[n])
    else:
        node, available_constraints = random.choice(list(available_edges.items()))
        c = random.choice(available_constraints)

    return (node, c)


def choose_weight() -> float:
    w = 0
    while w == 0:
        w = random.uniform(0, 1)

    return w


def add_edge(
    n: int,
    c: Tuple[str, str],
    available: Dict[int, List[Tuple[str, str]]],
    edges: Dict[Tuple[str, str], List[int]],
    arity: int,
):
    edges[c].append(n)
    available[n].remove(c)
    # if a constraint has reached the max arity, it can't be chosen anymore
    if len(edges[c]) == arity:
        to_be_removed = list()
        for node, l in available.items():
            if node != n and c in l:
                l.remove(c)
            if not l:
                # Can't remove it while iterating
                to_be_removed.append(node)
        for n in to_be_removed:
            available.pop(n)

    # In case a node has removed the last constraint from its available
    # constraints, but this constraint has not maximum arity, we still have
    # to remove this node from the possible future edges
    if n in available and not available[n]:
        available.pop(n)


def write_in_file(filename: str, dcop_str: str):
    path = "/".join(filename.split("/")[:-1])

    if (path != "") and (not os.path.exists(path)):
        os.makedirs(path)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(dcop_str)


def find_objective(weights: List[float], n: int, is_hard: bool):

    objective = 0
    if is_hard:
        # Choose an objective which is reachable
        for i in range(len(weights)):
            objective += random.choice(range(n)) * weights[i]
    else:
        objective = random.uniform(0, len(weights) * n)

    return objective


def correct_density(filename: str, real_density: float):
    path_elts = filename.split("/")
    for i in range(len(path_elts)):
        if path_elts[i].split("=")[0] == "density":
            path_elts[i] = "density={}".format(round(real_density, 1))

    return "/".join(path_elts)


def generate_ising(args):
    domain_size = args.range
    size = args.size
    d = VariableDomain("d", "dummy", range(domain_size))
    variables = {}
    constraints = {}
    for i in range(size):
        for j in range(size):
            v = Variable("v{}_{}".format(i, j), d, floor(domain_size * random.random()))
            variables[(i, j)] = v

    for i, j in variables:
        c = _create_ising_constraint(i, j, i, (j + 1) % size, domain_size, variables)
        constraints[(i, j, i, (j + 1) % size)] = c

        c = _create_ising_constraint(i, j, (i + 1) % size, j, domain_size, variables)
        constraints[(i, j, (i + 1) % size), j] = c

    dcop = DCOP("radom ising", "min")
    # dcop.domains = {'d': d}
    # dcop.variables = variables
    dcop._agents_def = {}
    for c in constraints.values():
        dcop.add_constraint(c)

    if args.output:
        outputfile = args.output[0]
        write_in_file(outputfile, dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


def _create_ising_constraint(i, j, i1, j1, domain_size, variables):
    target = floor(domain_size * random.random())
    v = "v{}_{}".format(i, j)
    v1 = "v{}_{}".format(i1, j1)
    c = relation_from_str(
        "c_{}_{}_{}_{}".format(i, j, i1, j1),
        "{} + {} - {}".format(v, v1, target),
        [variables[(i, j)], variables[i1, j1]],
    )
    return c
