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
.. _pydcop_commands_generate_graphcoloring:

pydcop generate graphcoloring
=============================

Graph coloring benchmark problem generator
------------------------------------------


Synopsis
--------

::

  pydcop generate graphcoloring
                --variables_count <variables_count>
                --colors_count <colors_count>
                --graph <graph_type>
                [--allow_subgraph]
                [--soft]
                [--extensive]
                [--noagents]
                [--p_edge <p_edge>]
                [--m_edge <m_edge>]


Description
-----------

This command generates graph coloring problems, soft or hard,
on several types of graphs.

Graph structures:

* Random graph, based on the Erdős-Rényi model
* Preferential attachment graph (scale free), based on the Barabási–Albert model
* Grid graph

Problems:

* Graph coloring, with pseudo-hard constraint where a cost of 10000 is incurred
  for neighbors sharing the same color and 0 otherwise. These constraints can
  be expressed intentionally or extensively.
* Weighted, or soft graph coloring, with soft constraints where the cost of a
  join assignment between two neighbors is a random cost, always expressed
  extensively.


**Note:** the generated DCOP is written to the standard output. To write it in a file,
you can use the ``--output <file>`` :ref:`global option<usage_cli_ref_options>`.

Options
-------

``--variables_count <variables_count>`` / ``-v <variables_count>``
  Number of variables in the problem. If the grid graph structure is used, this
  number MUST be a valid square side (i.e. :math:`\sqrt{v}` must be an ``int``).

``--colors_count <colors_count>`` / ``-c <colors_count>``
  Number of colors for coloring the graph.

``--graph <graph_type>`` / ``--g <graph_type>``
  Structure of the constraints graph. Can be ``random``, ``scalefree`` or
  ``grid``.

``--allow_subgraph``
  When generating the graph structure, we only keep graph with no disconnected
  sub-graph. When this flag is set no filtering is done and you may get
  disconnected sub-graphs.

``--soft``
  If this flag is set, a weighted graph coloring problem is generated, otherwise
  a classical (pseudo-) hard graph coloring problem is generated.

``--intentional``
  If this flag is set, constraints are expressed intentionally, otherwise an
  extensive representation is used (by default). Intentional constraints can
  only be used for standard graph coloring problems.

``--noagents``
  If this flag is set, no agent definition is generated in the dcop file,
  otherwise one agent is created for each variable.

``--p_edge <p_edge>`` / ``--p <p_edge>``
  Only used for random graph, probability for edge creation in the random
  Erdős-Rényi graph creation model.

``--m_edge <m_edge>`` / ``--m <m_edge>``
  Only used for scale-free graph, number of edges to attach from a new variable
  in the preferential attachment Barabási–Albert graph model





Examples
--------

Generating a random soft graph coloring problem with 10 variables::

    pydcop generate graph_coloring --graph random  --variables_count 10 \\
        --colors_count 3  --p_edge 0.5 --soft



"""
import logging
import math
import random

import networkx as nx

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import VariableDomain, Variable, AgentDef
from pydcop.dcop.relations import relation_from_str, NAryMatrixRelation
from pydcop.dcop.yamldcop import dcop_yaml

logger = logging.getLogger("pydcop.cli.generate")

COLORS = ["R", "G", "B", "O", "F", "Y", "L", "C"]


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser(
        "graph_coloring", help="Generate a graph coloring benchmark problem"
    )
    parser.set_defaults(func=generate)

    parser.add_argument(
        "-v", "--variables_count", type=int, required=True, help="Number of variables"
    )

    parser.add_argument(
        "-c",
        "--colors_count",
        required=True,
        type=int,
        help="Number of colors in the problem",
    )

    parser.add_argument(
        "-g",
        "--graph",
        required=True,
        choices=["random", "grid", "scalefree"],
        help="type of network graph: random, grid or scalefree",
    )

    parser.add_argument(
        "--allow_subgraph",
        action="store_true",
        default=False,
        help="allows (but does not guarantee) the graph to "
        "have several disconnected subgraphs, "
        "by default we generate problem with a single "
        "connected graph",
    )

    parser.add_argument(
        "--soft",
        default=False,
        required=False,
        action="store_true",
        help="Generate a weighted graph coloring problem",
    )

    parser.add_argument(
        "--intentional",
        default=False,
        required=False,
        action="store_true",
        help="generate the problem in intentional form (default " "is extensive form)",
    )

    parser.add_argument(
        "--noagents",
        default=False,
        required=False,
        action="store_true",
        help="Do not generate agents",
    )

    # For random graphs
    parser.add_argument(
        "-p",
        "--p_edge",
        required=False,
        type=float,
        default=None,
        help="Probability for edge creation, only used for "
        "random Erdős-Rényi graphs",
    )

    # For scale-free / preferential attachment graph
    parser.add_argument(
        "-m",
        "--m_edge",
        required=False,
        type=int,
        default=None,
        help="Number of edges to attach from a new variable "
        "to existing variables, only used for "
        "Erdős-Rényi graphs",
    )


def generate(args):
    """
    Generate and output a graph coloring problem
    """
    if args.colors_count > len(COLORS):
        raise ValueError("Too many colors!")

    if args.graph == "random":
        if not args.p_edge:
            raise ValueError(
                "Option --p_edge is mandatory when generating a graph coloring "
                "problem based on a random graph."
            )
        graph = generate_random_graph(
            args.variables_count, args.p_edge, args.allow_subgraph
        )
        name = "Random "
    elif args.graph == "scalefree":
        if not args.m_edge:
            raise ValueError(
                "Option --m_edge is mandatory when generating a graph coloring "
                "problem based on a barabasi graph."
            )
        graph = generate_scalefree_graph(
            args.variables_count, args.m_edge, args.allow_subgraph
        )
        name = "Scale-free "
    elif args.graph == "grid":
        graph = generate_grid_graph(args.variables_count)
        name = "Grid"
    else:
        raise ValueError("Invalid graph type for graphcoloring: " + args.graph)

    domain = VariableDomain("colors", "color", COLORS[: args.colors_count])

    variables = {}
    for i, node in enumerate(sorted(graph.nodes)):
        logger.debug("node %s - %s", node, i)
        name = f"v{i:02d}"
        # Networkx's nodes may be index or tuple, but are guaranteed to be
        # hashable, we can use them as key in our map:
        variables[node] = Variable(name, domain)

    agents = {}
    if not args.noagents:
        for i, _ in enumerate(variables):
            agt = AgentDef(f"a{i:02d}")
            agents[agt.name] = agt

    if args.soft:
        constraints = generate_soft_constraints(graph, variables, args.intentional)
        name += "soft graph coloring"
    else:
        constraints = generate_hard_constraints(graph, variables, args.intentional)
        name += "hard graph coloring"

    dcop = DCOP(
        name,
        domains={"colors": domain},
        variables={v.name: v for v in variables.values()},
        agents=agents,
        constraints=constraints,
    )

    if args.output:
        output_file = args.output
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


def generate_random_graph(variables_count, p_edge, allow_subgraph):
    if not allow_subgraph:
        graph = nx.gnp_random_graph(variables_count, p_edge)
        is_connected = nx.is_connected(graph)
        while not is_connected:
            graph = nx.gnp_random_graph(variables_count, p_edge)
            is_connected = nx.is_connected(graph)
    else:
        graph = nx.gnp_random_graph(variables_count, p_edge)
    return graph


def generate_scalefree_graph(variables_count, m_edge, allow_subgraph):
    if not allow_subgraph:
        graph = nx.barabasi_albert_graph(variables_count, m_edge)
        is_connected = nx.is_connected(graph)
        while not is_connected:
            graph = nx.barabasi_albert_graph(variables_count, m_edge)
            is_connected = nx.is_connected(graph)
    else:
        graph = nx.barabasi_albert_graph(variables_count, m_edge)

    # In the obtained graph, low rank nodes will have a much higher edge count
    # than high rank nodes. We shuffle the nodes names to avoid this effect:
    new_nodes = list(range(variables_count))
    random.shuffle(new_nodes)
    node_mapping = {n: nn for n, nn in zip(graph.nodes, new_nodes)}

    new_graph = nx.Graph((node_mapping[e1], node_mapping[e2]) for e1, e2 in graph.edges)
    return new_graph


def generate_grid_graph(variables_count):
    side = math.sqrt(variables_count)
    if int(side) != side:
        raise ValueError(
            f"The value {variables_count} provided for"
            "the option --variables_count is not a valid square"
            "grid size"
        )
    side = int(side)
    graph = nx.grid_2d_graph(side, side)
    return graph


def generate_soft_constraints(graph, variables, intentional):
    constraints = {}
    if intentional:
        raise ValueError(
            "Cannot generate soft intentional " "graph coloring constraints"
        )
    for i, edge in enumerate(graph.edges):
        logger.debug("edge %s - %s", edge, i)
        name = "c" + str(i)
        u, v = edge
        v1, v2 = variables[u], variables[v]
        constraint = NAryMatrixRelation([v1, v2], name=name)
        for val1 in v1.domain:
            for val2 in v2.domain:
                constraint = constraint.set_value_for_assignment(
                    {v1.name: val1, v2.name: val2}, random.randint(0, 9)
                )
        constraints[name] = constraint
        logger.debug(repr(constraints[name]))

    return constraints


def generate_hard_constraints(graph, variables, intentional):
    """
    Generate a hard constraint for each edge in the graph
    Parameters
    ----------
    graph:
        a networkx graph representing the constraint network
    variables: dict
        a dict of variable objects
    intentional: bool
        if true, generate intentional constraints

    Returns
    -------
    dict:
        a dict of constraints
    """
    constraints = {}
    for i, edge in enumerate(graph.edges):
        logger.debug("edge %s - %s", edge, i)
        name = "c" + str(i)
        u, v = edge
        v1, v2 = variables[u], variables[v]
        if intentional:
            expression = f"1000 if {v1.name} == {v2.name} else 0"
            constraints[name] = relation_from_str(name, expression, [v1, v2])
        else:
            constraint = NAryMatrixRelation([v1, v2], name=name)
            for val in v1.domain:
                constraint = constraint.set_value_for_assignment(
                    {v1.name: val, v2.name: val}, 1000
                )
            constraints[name] = constraint
        logger.debug(repr(constraints[name]))

    return constraints
