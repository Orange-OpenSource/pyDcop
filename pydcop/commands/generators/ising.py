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
.. _pydcop_commands_generate_ising:


pydcop generate ising
=====================

Ising benchmark problem generator
---------------------------------

::

  pydcop generate ising
                --row_count <row_count>
                [--col_count <col_count>]
                [--bin_range <bin_range>]
                {--un_range <un_range>]
                [-intentional]
                [--fg_dist]


Description
-----------

This command generates a DCOP modelling an ising problem :cite:`cerquides_improving_2018`.

The Ising model is a widely used benchmark in statistical physics ; constraint graphs
are rectangular grids where each binary variable is connected to its four closer neighbors
(with toroidal links which connect opposite sides of the grid), and is constrained by a unary cost
:math:`r_i`.

The weight of each binary constraint
:math:`r_{ij}` is determined by first sampling a value
:math:`k_{ij}` from a uniform distribution :math:`U[−\\beta,\\beta]`
and then assigning
:math:`r_{ij}(x_i,x_j) = k_{ij}` if :math:`x_i = x_j, −κ_{ij}` otherwise.
The :math:`\\beta` parameter controls the average strength of interactions.

The weight for each unary constraint
:math:`r_i` is determined by sampling :math:`k_i`
from a uniform distribution :math:`U[-\\rho, \\rho]`
and then assigning :math:`r_i (0) = k_i` and :math:`r_i(1) = −k_i`.

When using ``--var_dist``, this generator produces one agent for each variable and a
distribution file that maps each variable to one agent.
When using a factor-graph based algorithm, you can use the ``--fg_dist`` flag in
order to generate a distribution that maps one variable and 3 constraints to each agents.
Each unary constraint is mapped to the agent holding the corresponding variable.
This agent also takes responsibility for the binary constraints
on the right and bellow this variable in the grid.
Both options can be used simultaneously, in which case both distributions will be
generated.

**Note:** the generated DCOP and distribution(s) are written to the standard output.
To write them in files,
you can use the ``--output <file>`` :ref:`global option<usage_cli_ref_options>`.


Options
-------

``--row_count <row_count>``
  Number of rows in the grid, must be >= 2.

``--col_count <col_count>``
  Number of columns in the grid, optional. If ``col_count`` is not given, the generated
  will be a square of size ``row_count``. If given, ``col_count`` must be >= 2.

``--bin_range <bin_range>``
  :math:`\\beta` value used for binary constraints. Defaults to 1.6.

``--un_range <un_range>``
  :math:`\\rho` value used for unary constraints. Defaults to 0.05.

``--intentional``
  When using this flag, constraints are generated in the intentional form
  (default is extensive).

``--fg_dist``
  When using this flag, the agents and distribution are generated for factor-graph
  based algorithms where computations are needed for variables and constraints.
  When outputting (with the ``--output`` global option) the dcop in a file
  ``<dcop_name.yaml>``,  the distribution is automatically written to a file
  ``<dcop_name>_fgdist.yaml``.

``--var_dist``
  When using this flag, the agents and distribution are generated for a classic
  constraint graph where computations are needed for variables and each agent is
  responsible one variable. When outputting (with the ``--output`` global option)
  the dcop in a file ``<dcop_name.yaml>``,  the distribution is automatically written
  to a file ``<dcop_name>_vardist.yaml``.


Examples
--------

Generate a DCOP representing a 3x4 ising problem, in extensive form::

    pydcop generate ising --row_count 3 --col_count 4


Generate a DCOP representing a 3x4 ising problem, in intentional form and written to disk::


    pydcop --output ising.yaml  generate ising --row_count 3 --col_count 4 \\
             --bin_range 1.6 --un_range 0.05 --intentional --fg_dist --var_dist


"""
import logging
import random
from collections import defaultdict
from os.path import splitext
from typing import Any, Dict, Tuple

import networkx as nx
import yaml

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable, Domain, AgentDef
from pydcop.dcop.relations import NAryMatrixRelation, Constraint, constraint_from_str
from pydcop.dcop.yamldcop import dcop_yaml

logger = logging.getLogger("pydcop.cli.generate")


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser(
        "ising", help="Generate an ising benchmark problem"
    )
    parser.set_defaults(func=generate)

    parser.add_argument(
        "--row_count", required=True, type=int, help="Number of rows in the grid"
    )
    parser.add_argument(
        "--col_count", required=False, type=int, help="Number of columns in the grid"
    )

    parser.add_argument(
        "--bin_range", type=float, default=1.6, help="Range of binary constraints"
    )
    parser.add_argument(
        "--un_range", type=float, default=0.05, help="Range of unary constraints"
    )
    parser.add_argument(
        "--intentional",
        default=False,
        required=False,
        action="store_true",
        help="generate the problem in intentional form (default is extensive form)",
    )
    parser.add_argument(
        "--no_agents",
        default=False,
        required=False,
        action="store_true",
        help="generate the problem without any agents. You can use the 'pydcop generate " \
             "agents' to generate them with their hosting and route costs"
    )

    parser.add_argument(
        "--fg_dist",
        default=False,
        required=False,
        action="store_true",
        help="When using this flag, the agents and distribution are generated for "
        "factor-graph based algorithms where computations are needed for variables "
        "and constraints.",
    )
    parser.add_argument(
        "--var_dist",
        default=False,
        required=False,
        action="store_true",
        help="When using this flag, the agents and distribution are generated for "
        "a classic constraint graph where computations are needed for variables and "
        "each agent is responsible one variable.factor-graph based algorithms where "
        "computations are needed for variables and constraints.",
    )


def generate(args):

    # Some extra checks on cli parameters!
    if args.row_count <= 2:
        raise ValueError("--row_count: The size must be > 2")
    if args.col_count:
        if args.col_count <= 2:
            raise ValueError("--col_count: The size must be > 2")
        col_count = args.col_count
    else:
        col_count = args.row_count

    dcop, var_mapping, fg_mapping = generate_ising(
        args.row_count,
        col_count,
        args.bin_range,
        args.un_range,
        not args.intentional,
        no_agents=args.no_agents,
        fg_dist=args.fg_dist,
        var_dist=args.var_dist,
    )

    graph = "factor_graph" if args.fg_dist else "constraints_graph"
    output_file = args.output if args.output else "NA"
    dist_result = {
        "inputs": {
            "dist_algo": "NA",
            "dcop": output_file,
            "graph": graph,
            "algo": "NA",
        },
        "cost": None,
    }

    # TODO: generate and output distribution
    if args.output:
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(dcop_yaml(dcop))
        path, ext = splitext(output_file)
        if args.fg_dist:
            dist_result["distribution"] = fg_mapping
            dist_output_file = f"{path}_fgdist{ext}"
            with open(dist_output_file, encoding="utf-8", mode="w") as fo:
                fo.write(yaml.dump(dist_result))
        if args.var_dist:
            dist_result["distribution"] = var_mapping
            dist_output_file = f"{path}_vardist{ext}"
            with open(dist_output_file, encoding="utf-8", mode="w") as fo:
                fo.write(yaml.dump(dist_result))

    else:
        print(dcop_yaml(dcop))
        if args.fg_dist:
            dist_result["distribution"] = fg_mapping
            print(yaml.dump(dist_result))
        if args.var_dist:
            dist_result["distribution"] = fg_mapping
            print(yaml.dump(dist_result))


def generate_ising(
    row_count: int,
    col_count: int,
    bin_range: float,
    un_range: float,
    extensive: bool,
    no_agents: bool,
    fg_dist: bool,
    var_dist: bool,
) -> Tuple[DCOP, Dict, Dict]:

    grid_graph = nx.grid_2d_graph(row_count, col_count, periodic=True)
    domain = Domain("var_domain", "binary", [0, 1])

    variables = generate_binary_variables(grid_graph, domain)

    constraints = {}
    unary_constraints = generate_unary_constraints(variables, un_range, extensive)
    constraints.update(unary_constraints)
    binary_constraints = generate_binary_constraints(
        grid_graph, variables, bin_range, extensive
    )
    constraints.update(binary_constraints)

    agents = {}
    fg_mapping = defaultdict(lambda: [])
    var_mapping = defaultdict(lambda: [])
    for (row, col) in grid_graph.nodes:
        agent = AgentDef(f"a_{row}_{col}")
        agents[agent.name] = agent
        left = (row - 1) % row_count
        down = (col + 1) % col_count

        if var_dist:
            var_mapping[agent.name].append(f"v_{row}_{col}")

        if fg_dist:
            fg_mapping[agent.name].append(f"v_{row}_{col}")
            fg_mapping[agent.name].append(f"cu_v_{row}_{col}")
            # Sort coordinate to make sure we build the name in the same order as when
            # creating the constraints:
            (r1, c1), (r2, c2) = sorted([(row, col), (left, col)])
            fg_mapping[agent.name].append(f"cb_v_{r1}_{c1}_v_{r2}_{c2}")
            (r1, c1), (r2, c2) = sorted([(row, col), (row, down)])
            fg_mapping[agent.name].append(f"cb_v_{r1}_{c1}_v_{r2}_{c2}")

    name = f"Ising_{row_count}_{col_count}_{bin_range}_{un_range}"
    if no_agents:
        agents = {}
    dcop = DCOP(
        name,
        domains={"var_domain": domain},
        variables={v.name: v for v in variables.values()},
        agents=agents,
        constraints=constraints,
    )

    return dcop, dict(var_mapping), dict(fg_mapping)


def generate_binary_variables(grid_graph: nx.Graph, domain: Domain):
    variables = {}
    for (row, col) in grid_graph.nodes:
        # Networkx's nodes are tuple when dealing with a grid graph.
        variable = Variable(f"v_{row}_{col}", domain)
        variables[variable.name] = variable
    return variables


def generate_binary_constraints(
    grid_graph: nx.Graph, variables, bin_range: float, extensive: bool
) -> Dict[str, Constraint]:
    constraints: Dict[str, Constraint] = {}
    for nodes in grid_graph.edges:
        (r1, c1), (r2, c2) = sorted(nodes)
        name1 = f"v_{r1}_{c1}"
        name2 = f"v_{r2}_{c2}"
        v1 = variables[name1]
        v2 = variables[name2]

        if extensive:
            constraint = generate_binary_extensive_constraint(v1, v2, bin_range)
        else:
            constraint = generate_binary_intentional_constraint(v1, v2, bin_range)
        constraints[constraint.name] = constraint
    return constraints


def generate_binary_extensive_constraint(
    variable1: Variable, variable2: Variable, bin_range: float
) -> Constraint:

    constraint = NAryMatrixRelation(
        [variable1, variable2], name=f"cb_{variable1.name}_{variable2.name}"
    )
    value = random.uniform(-bin_range, bin_range)
    constraint = constraint.set_value_for_assignment(
        {variable1.name: 0, variable2.name: 0}, value
    )
    constraint = constraint.set_value_for_assignment(
        {variable1.name: 1, variable2.name: 1}, value
    )
    constraint = constraint.set_value_for_assignment(
        {variable1.name: 0, variable2.name: 1}, -value
    )
    constraint = constraint.set_value_for_assignment(
        {variable1.name: 1, variable2.name: 0}, -value
    )
    return constraint


def generate_binary_intentional_constraint(
    variable1: Variable, variable2: Variable, bin_range: float
) -> Constraint:
    value = random.uniform(-bin_range, bin_range)

    constraint = constraint_from_str(
        name=f"cb_{variable1.name}_{variable2.name}",
        expression=f"{value} if {variable1.name} == {variable2.name} else -{value}",
        all_variables=[variable1, variable2],
    )

    return constraint


def generate_unary_constraints(
    variables: Dict[Any, Variable], un_range: float, extensive: bool
) -> Dict[str, Constraint]:
    constraints: Dict[str, Constraint] = {}
    for variable in variables.values():
        if extensive:
            constraint = generate_unary_extensive_constraint(variable, un_range)
        else:
            constraint = generate_unary_intentional_constraint(variable, un_range)
        constraints[constraint.name] = constraint
    return constraints


def generate_unary_extensive_constraint(
    variable: Variable, un_range: float
) -> Constraint:

    constraint = NAryMatrixRelation([variable], name=f"cu_{variable.name}")
    value = random.uniform(-un_range, un_range)
    constraint = constraint.set_value_for_assignment({variable.name: 0}, value)
    constraint = constraint.set_value_for_assignment({variable.name: 1}, -value)
    return constraint


def generate_unary_intentional_constraint(variable: Variable, un_range: float):
    value = random.uniform(-un_range, un_range)
    constraint = constraint_from_str(
        name=f"cu_{variable.name}",
        expression=f" -{value} if {variable.name} == 1 else {value}",
        all_variables=[variable],
    )
    return constraint
