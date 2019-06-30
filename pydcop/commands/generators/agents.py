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
.. _pydcop_commands_generate_agents:

pydcop generate agents
======================

Generate a set of agents, with their capacity and optionaly their hosting and
communication costs.



Synopsis
--------
::

    pydcop generate agents --mode <generation_mode>
                      --capacity <capacity>
                      [--count <agents_count>]
                      [--dcop_files <dcop files>]
                      [--agent_prefix <agent_prefix>]
                      [--hosting <hosting_cost_mode>]
                      [--hosting_default <hosting_default>]
                      [--routes_default <routes_default>]

Description
-----------

The agents can be generated for a given optimization problem or by simply specifying a
numberfof agents.


Options
-------
``--mode <generation_mode>``
  Agents generation mode. When 'variables' is used, one agent is generated for each
  variable in the problem and and the '--dcop_files' option is required.
  When using 'count', the '--count' option is required.

``--capacity <capacity>``
  The capacity of agents. All agents will have the same capacity.

``--count <agents_count>``
  Number of agents to generate. Must be given when using the ``--mode count`` option.

``--dcop_files <dcop files>``
  The problem agents are created for. It can be given as one or several files, which
  content will be appended before parsing.

``--agent_prefix <agent_prefix>``
  The prefix to use when generating agent's name. default to "a".

``--hosting <hosting_cost_mode>``
  Mode of generation for hosting costs, one of ``None``, ``name_mapping`` or
  ``var_startswith``. When using ``name_mapping`` a 0 hosting cost will be generated
  for computations that have the same name as the agent (excluding prefix, which are
  automatically detected). With ``var_startswith`` the mapping of variables to agent
  will only consider the start of the variable name (still excluding prefix).

``--hosting_default <hosting_default>``
  Default hosting cost, mandatory when using ``--hosting name_mapping``

``--routes <route_cost_mode>``
  Mode of generation for route costs, one of ``None``, ``uniform`` or ``graph``.
  When using ``uniform``, all routes have the same cost, given with ``-routes_default``

``--routes_default <routes_default>``
  Default route cost

Examples
========

Simply generate 10 agents with a 100 capacity. Note that we do not pass a DCOP
file in that case::

  pydcop generate agents --mode count --count 10 --capacity 100

Generate agents, one for each variable, and hosting costs::

  pydcop generate agents --mode variables --capacity 100 --dcop_files ising_dcop.yaml \
      --hosting "name_mapping" --hosting_default 200



"""
import logging
import re
from collections import defaultdict
from typing import List, Dict

from pydcop.computations_graph import constraints_hypergraph
from pydcop.dcop.objects import AgentDef
from pydcop.dcop.yamldcop import yaml_agents, load_dcop_from_file

logger = logging.getLogger("pydcop.cli.generate")


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser("agents", help="Generate a set of agents")
    parser.set_defaults(func=generate)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["variables", "count"],
        help="Agents generation mode. When 'variables' is used, one agent "
        "is generated for each agent and the '--dcop_files' option is required. "
        "When using 'count', '--count' is required",
    )

    parser.add_argument("--dcop_files", type=str, nargs="+", default=None, help="dcop file(s)")

    parser.add_argument("--count", type=int, help="Number of agents")

    parser.add_argument(
        "--agent_prefix", type=str, default="a", help="Prefix when creating agent"
    )

    parser.add_argument(
        "--capacity", type=int, required=True, help="Capacity of agents"
    )

    parser.add_argument(
        "--hosting",
        choices=["None", "name_mapping", "var_startswith"],
        required=False,
        default="None",
        help="Hosting cost generation method.",
    )
    parser.add_argument(
        "--hosting_default",
        type=int,
        required=False,
        help="Default hosting cost, mandatory when using --hosting",
    )

    parser.add_argument(
        "--routes",
        choices=["None", "uniform", "graph"],
        required=False,
        default="None",
        help="Route cost generation method.",
    )
    parser.add_argument(
        "--routes_default",
        type=int,
        required=False,
        help="Default routes cost, mandatory when using --routes",
    )

    parser.add_argument(
        "dcop_files_end", type=str, nargs="*", metavar="FILE", 
        help="dcop file(s)", default=None,
    )

def generate(args):
    if not args.dcop_files and args.dcop_files_end:
        args.dcop_files = args.dcop_files_end

    check_args(args)

    variables = []
    if args.dcop_files:
        logger.info("loading dcop from {}".format(args.dcop_files))
        dcop = load_dcop_from_file(args.dcop_files)
        variables = list(dcop.variables)

    agents_name = generate_agents_names(
        args.mode, args.count, variables, args.agent_prefix
    )

    mapping = {}
    hosting_costs = {}
    if args.hosting != "None":
        mapping = agent_variables_mapping(args.hosting, agents_name, variables)
        hosting_costs = generate_hosting_costs(args.hosting, mapping)

    routes_costs = {}
    if args.routes != "None":
        routes_costs = generate_routes_costs(args.routes, mapping, dcop)

    agents = []
    for agt_name in agents_name:
        kw = {}
        if agt_name in hosting_costs:
            kw["hosting_costs"] = hosting_costs[agt_name]
        if args.hosting_default:
            kw["default_hosting_cost"] = args.hosting_default
        if args.capacity:
            kw["capacity"] = args.capacity
        if agt_name in routes_costs:
            kw["routes"] = routes_costs[agt_name]
        if args.routes_default:
            kw["default_route"] = args.routes_default
        agents.append(AgentDef(agt_name, **kw))

    serialized = yaml_agents(agents)

    if args.output:
        output_file = args.output
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(serialized)
    else:
        print(serialized)


def check_args(args):
    if args.mode == "count" and not args.count:
        raise ValueError(
            "--count is required when using 'count' agents generation mode"
        )

    if args.mode == "variables" and not args.dcop_files:
        raise ValueError(
            "--dcop_files is required when using 'variables' agents generation mode"
        )
    if args.hosting:
        if args.hosting != "None" and not args.dcop_files:
            raise ValueError(
                f"Missing dcop file when using {args.hosting} hosting cost generation"
            )
        if args.hosting != "None" and not args.hosting_default:
            raise ValueError(
                "--hosting_default is mandatory when using --hosting cost generation"
            )
    if args.routes:
        if args.routes != "None" and not args.routes_default:
            raise ValueError(
                "--routes_default is mandatory when using --routes cost generation"
            )


def generate_agents_names(
    mode: str, count=None, variables=None, agent_prefix="a"
) -> List[str]:
    if mode == "count":
        return generate_agents_from_count(count, agent_prefix=agent_prefix)
    elif mode == "variables":
        return generate_agents_from_variables(variables, agent_prefix=agent_prefix)
    raise ValueError(f"Invalid mode {mode}")


def generate_agents_from_count(agent_count: int, agent_prefix="a") -> List[str]:
    digit_count = len(str(agent_count - 1))
    agents = [f"{agent_prefix}{i:0{digit_count}d}" for i in range(agent_count)]
    return agents


def generate_agents_from_variables(variables: List[str], agent_prefix="a") -> List[str]:
    prefix_length = len(find_prefix(variables))

    return [agent_prefix + variable[prefix_length:] for variable in variables]


def agent_variables_mapping(
    hosting_mode: str, agents: List[str], variables: List[str]
) -> Dict[str, List[str]]:
    if hosting_mode == "name_mapping":
        return find_corresponding_variables(agents, variables)
    elif hosting_mode == "var_startswith":
        return find_corresponding_variables_start_with(agents, variables)


def generate_hosting_costs(mode: str,  mapping: Dict[str, List[str]]):

    costs = {}
    for agt_name in mapping:
        agt_costs = {}
        for var_name in mapping[agt_name]:
            agt_costs[var_name] = 0
        costs[agt_name] = agt_costs
    return costs


def generate_routes_costs(
    mode: str, mapping, dcop
) -> Dict[str, Dict[str, float]]:
    routes = {}
    if mode == "graph":
        variables = list(dcop.variables)
        graph = constraints_hypergraph.build_computation_graph(dcop)

        # route = (1 + abs(degree_n - degree_v)) / (degree_n + degree_v)
        # logger.debug(f"agants {agents}")
        # logger.debug(f"variables {variables}")
        # mappings = find_corresponding_variables(agents, variables)
        # logger.debug(mappings)
        inverse_mapping = {
            variable: agent
            for agent, variables in mapping.items()
            for variable in variables
        }
        logger.debug(inverse_mapping)
        for agt_name in mapping:
            agt_routes = {}
            for hosted_variable in mapping[agt_name]:
                # hosted_variable = mappings[agt_name]
                neighbor_variables = list(graph.neighbors(hosted_variable))
                logger.debug(
                    f"routes for {agt_name} hosting {hosted_variable} with {neighbor_variables}"
                )

                degree = len(neighbor_variables)
                for neighbor in neighbor_variables:
                    if neighbor in inverse_mapping:
                        neighbor_agent = inverse_mapping[neighbor]
                        degree_neighbor = len(list(graph.neighbors(neighbor)))
                        route = (0.2 + abs(degree - degree_neighbor)) / (
                            degree + degree_neighbor
                        )
                        logger.debug(
                            f"Route {agt_name} - {neighbor_agent} : {route}"
                        )
                        agt_routes[neighbor_agent] = route
            routes[agt_name] = agt_routes
    return routes


def find_corresponding_variables(
    agents: List[str], variables: List[str], agt_prefix=None, var_prefix=None
) -> Dict[str, List[str]]:
    var_prefix = var_prefix if var_prefix else find_prefix(variables)
    var_regexp = re.compile(f"{var_prefix}(?P<index_var>\w+)")
    agt_prefix = agt_prefix if agt_prefix else find_prefix(agents)
    agt_regexp = re.compile(f"{agt_prefix}(?P<index_agt>\w+)")

    mapping, indexed_vars = defaultdict(lambda: list()), {}
    for variable in variables:
        m = var_regexp.match(variable)
        if m:
            index = m.group("index_var")
            indexed_vars[index] = variable

    try:
        int_indexed_vars = {
            int(index_var): variable for index_var, variable in indexed_vars.items()
        }
        use_int_index = True
    except ValueError:
        int_indexed_vars = []

    for agent in agents:
        m = agt_regexp.match(agent)
        if m:
            index = m.group("index_agt")
            if index in indexed_vars:
                mapping[agent].append(indexed_vars[index])
            elif use_int_index:
                # Try with int index with str index could not be found
                try:
                    index = int(index)
                    if index in int_indexed_vars:
                        mapping[agent].append(int_indexed_vars[index])
                except ValueError:
                    pass

    return dict(mapping)


def find_corresponding_variables_start_with(
    agents: List[str], variables: List[str], agt_prefix=None, var_prefix=None
) -> Dict[str, List[str]]:

    var_prefix = var_prefix if var_prefix else find_prefix(variables)
    var_regexp = re.compile(f"{var_prefix}(?P<index_var>\w+)")
    agt_prefix = agt_prefix if agt_prefix else find_prefix(agents)
    agt_regexp = re.compile(f"{agt_prefix}(?P<index_agt>\w+)")

    mapping, indexed_vars = defaultdict(lambda: list()), {}
    for variable in variables:
        m = var_regexp.match(variable)
        if m:
            index = m.group("index_var")
            indexed_vars[index] = variable

    try:
        int_indexed_vars = {
            int(index_var): variable for index_var, variable in indexed_vars.items()
        }
    except ValueError:
        int_indexed_vars = []

    for agent in agents:
        m = agt_regexp.match(agent)
        if m:
            index = m.group("index_agt")
            var_startswith = [
                indexed_vars[root_var]
                for root_var in indexed_vars
                if root_var.startswith(index)
            ]
            mapping[agent].extend(var_startswith)

    return dict(mapping)


def find_prefix(names: List[str]) -> str:
    """
    Find a common prefix in a list of string?
    Parameters
    ----------
    names: list of str

    Returns
    -------
    prefix: str
    """
    prefix_lenght = 1
    prefix = ""
    while True:
        prefix_test = names[0][:prefix_lenght]
        if all(name[:prefix_lenght] == prefix_test for name in names):
            prefix_lenght += 1
            prefix = prefix_test
            continue
        break

    return prefix
