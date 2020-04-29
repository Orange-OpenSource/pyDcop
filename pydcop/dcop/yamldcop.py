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
import pathlib
from collections import defaultdict
from collections import Iterable as CollectionIterable
from typing import Dict, Iterable, Union, List

import yaml

from pydcop.dcop.objects import (
    VariableDomain,
    Variable,
    ExternalVariable,
    VariableWithCostFunc,
    VariableNoisyCostFunc,
    AgentDef,
)
from pydcop.dcop.scenario import EventAction, DcopEvent, Scenario
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import (
    relation_from_str,
    RelationProtocol,
    NAryMatrixRelation,
    assignment_matrix,
    generate_assignment_as_dict, constraint_from_str,
    constraint_from_external_definition,
)
from pydcop.utils.expressionfunction import ExpressionFunction
from pydcop.distribution.objects import DistributionHints


class DcopInvalidFormatError(Exception):
    pass


def load_dcop_from_file(filenames: Union[str, Iterable[str]]):
    """
    load a dcop from one or several files

    Parameters
    ----------
    filenames: str or iterable of str
        The dcop can the given as a single file or as several files. When
        passing an iterable of file names, their content is concatenated
        before parsing. This can be usefull when you want to define the
        agents in a separate file.

    Returns
    -------
    A DCOP object built by parsing the files

    """
    content = ""
    main_dir = None

    if not isinstance(filenames, CollectionIterable):
        filenames = [filenames]

    for filename in filenames:
        p = pathlib.Path(filename)
        if main_dir is None:
            main_dir = p.parent
        content += p.read_text(encoding="utf-8")

    if content:
        return load_dcop(content, main_dir)


def load_dcop(dcop_str: str, main_dir=None) -> DCOP:
    loaded = yaml.load(dcop_str, Loader=yaml.FullLoader)

    if "name" not in loaded:
        raise ValueError("Missing name in dcop string")
    if "objective" not in loaded or loaded["objective"] not in ["min", "max"]:
        raise ValueError("Objective is mandatory and must be min or max")

    dcop = DCOP(
        loaded["name"],
        loaded["objective"],
        loaded["description"] if "description" in loaded else "",
    )

    dcop.domains = _build_domains(loaded)
    dcop.variables = _build_variables(loaded, dcop)
    dcop.external_variables = _build_external_variables(loaded, dcop)
    dcop._constraints = _build_constraints(loaded, dcop, main_dir)
    dcop._agents_def = _build_agents(loaded)
    dcop.dist_hints = _build_dist_hints(loaded, dcop)
    return dcop


def dcop_yaml(dcop: DCOP) -> str:

    dcop_dict = {"name": dcop.name, "objective": dcop.objective}
    dcop_str = yaml.dump(dcop_dict, default_flow_style=False)
    dcop_str += "\n"
    dcop_str += _yaml_domains(dcop.domains.values())
    dcop_str += "\n"
    dcop_str += _yaml_variables(dcop.variables.values())
    dcop_str += "\n"
    dcop_str += _yaml_constraints(dcop.constraints.values())
    dcop_str += "\n"
    dcop_str += yaml_agents(dcop.agents.values())

    return dcop_str


def _yaml_domains(domains):
    d_dict = {}
    for domain in domains:
        d_dict[domain.name] = {"values": list(domain.values), "type": domain.type}
    return yaml.dump({"domains": d_dict})  #  , default_flow_style=False)


def _build_domains(loaded) -> Dict[str, VariableDomain]:
    domains = {}
    if "domains" in loaded:
        for d_name in loaded["domains"]:
            d = loaded["domains"][d_name]
            values = d["values"]

            if len(values) == 1 and ".." in values[0]:
                values = str_2_domain_values(d["values"][0])
            d_type = d["type"] if "type" in d else ""
            domains[d_name] = VariableDomain(d_name, d_type, values)

    return domains


def _yaml_variables(variables):
    var_dict = {}
    for v in variables:
        var_dict[v.name] = {"domain": v.domain.name}
        if v.initial_value is not None:
            var_dict[v.name]["initial_value"] = v.initial_value

    return yaml.dump({"variables": var_dict}, default_flow_style=False)


def _build_variables(loaded, dcop) -> Dict[str, Variable]:
    variables = {}
    if "variables" in loaded:
        for v_name in loaded["variables"]:
            v = loaded["variables"][v_name]
            domain = dcop.domain(v["domain"])
            initial_value = v["initial_value"] if "initial_value" in v else None
            if initial_value and initial_value not in domain.values:
                raise ValueError(
                    "initial value {} is not in the domain {} "
                    "of the variable {}".format(initial_value, domain.name, v_name)
                )

            if "cost_function" in v:
                cost_expression = v["cost_function"]
                cost_func = ExpressionFunction(cost_expression)
                if "noise_level" in v:
                    variables[v_name] = VariableNoisyCostFunc(
                        v_name,
                        domain,
                        cost_func,
                        initial_value,
                        noise_level=v["noise_level"],
                    )
                else:
                    variables[v_name] = VariableWithCostFunc(
                        v_name, domain, cost_func, initial_value
                    )

            else:
                variables[v_name] = Variable(v_name, domain, initial_value)
    return variables


def _build_external_variables(loaded, dcop) -> Dict[str, ExternalVariable]:
    ext_vars = {}
    if "external_variables" in loaded:
        for v_name in loaded["external_variables"]:
            v = loaded["external_variables"][v_name]
            domain = dcop.domain(v["domain"])
            initial_value = v["initial_value"] if "initial_value" in v else None
            if initial_value and initial_value not in domain.values:
                raise ValueError(
                    "initial value {} is not in the domain {} "
                    "of the variable {}".format(initial_value, domain.name, v_name)
                )
            ext_vars[v_name] = ExternalVariable(v_name, domain, initial_value)
    return ext_vars


def _build_constraints(loaded, dcop, main_dir) -> Dict[str, RelationProtocol]:
    constraints = {}
    if "constraints" in loaded:
        for c_name in loaded["constraints"]:
            c = loaded["constraints"][c_name]
            if "type" not in c:
                raise ValueError(
                    "Error in contraints {} definition: type is "
                    'mandatory and only "intention" is '
                    "supported for now".format(c_name)
                )
            elif c["type"] == "intention":
                if "source" in c:
                    src_path = c["source"] \
                        if pathlib.Path(c["source"]).is_absolute() \
                        else main_dir / c["source"]
                    constraints[c_name] = constraint_from_external_definition(
                        c_name, src_path, c["function"], dcop.all_variables
                    )
                else:
                    constraints[c_name] = constraint_from_str(
                        c_name, c["function"], dcop.all_variables
                    )
            elif c["type"] == "extensional":
                values_def = c["values"]
                default = None if "default" not in c else c["default"]
                if type(c["variables"]) != list:
                    # specific case for constraint with a single variable
                    v = dcop.variable(c["variables"].strip())
                    values = [default] * len(v.domain)
                    for value, assignments_def in values_def.items():
                        if isinstance(assignments_def, str):
                            for ass_def in assignments_def.split("|"):
                                iv, _ = v.domain.to_domain_value(ass_def.strip())
                                values[iv] = value
                        else:
                            values[v.domain.index(assignments_def)] = value

                    constraints[c_name] = NAryMatrixRelation([v], values, name=c_name)
                    continue

                # For constraints that depends on several variables
                vars = [dcop.variable(v) for v in c["variables"]]
                values = assignment_matrix(vars, default)

                for value, assignments_def in values_def.items():
                    # can be a str like "1 2 3" or "1 2 3 | 1 3 4"
                    # several assignment for the same value are separated with |
                    assignments_def = assignments_def.split("|")
                    for ass_def in assignments_def:
                        val_position = values
                        vals_def = ass_def.split()
                        for i, val_def in enumerate(vals_def[:-1]):
                            iv, _ = vars[i].domain.to_domain_value(val_def.strip())
                            val_position = val_position[iv]
                        # value for the last variable of the assignment
                        val_def = vals_def[-1]
                        iv, _ = vars[-1].domain.to_domain_value(val_def.strip())
                        val_position[iv] = value

                constraints[c_name] = NAryMatrixRelation(vars, values, name=c_name)

            else:
                raise ValueError(
                    "Error in contraints {} definition: type is  mandatory "
                    'and must be "intention" or "intensional"'.format(c_name)
                )

    return constraints


def _yaml_constraints(constraints: Iterable[RelationProtocol]):
    constraints_dict = {}
    for r in constraints:
        if hasattr(r, "expression"):

            constraints_dict[r.name] = {"type": "intention", "function": r.expression}
        else:
            # fallback to extensional constraint
            variables = [v.name for v in r.dimensions]
            values = defaultdict(lambda: [])

            for assignment in generate_assignment_as_dict(r.dimensions):
                val = r(**assignment)
                ass_str = " ".join([str(assignment[var]) for var in variables])
                values[val].append(ass_str)

            for val in values:
                values[val] = " | ".join(values[val])
            values = dict(values)
            constraints_dict[r.name] = {
                "type": "extensional",
                "variables": variables,
                "values": values,
            }

    return yaml.dump({"constraints": constraints_dict}, default_flow_style=False)


def _build_agents(loaded) -> Dict[str, AgentDef]:

    # Read agents list, without creating AgentDef object yet.
    # We need the preferences to create the AgentDef objects
    agents_list = {}
    if "agents" in loaded:
        for a_name in loaded["agents"]:
            try:
                kw = loaded["agents"][a_name]
                # we accept any attribute for the agent
                # Most of the time it will be capacity and also preference but
                # any named value is valid:
                agents_list[a_name] = kw if kw else {}
            except TypeError:
                # means agents are given as a list and not a map:
                agents_list[a_name] = {}

    routes = {}
    default_route = 1
    if "routes" in loaded:
        for a1 in loaded["routes"]:
            if a1 == "default":
                default_route = loaded["routes"]["default"]
                continue
            if a1 not in agents_list:
                raise DcopInvalidFormatError("Route for unknown " "agent " + a1)
            a1_routes = loaded["routes"][a1]
            for a2 in a1_routes:
                if a2 not in agents_list:
                    raise DcopInvalidFormatError("Route for unknown " "agent " + a2)
                if (a2, a1) in routes or (a1, a2) in routes:
                    if routes[(a2, a1)] != a1_routes[a2]:
                        raise DcopInvalidFormatError(
                            "Multiple route definition r{} = {}"
                            " != r({}) = {}".format(
                                (a2, a1), routes[(a2, a1)], (a1, a2), a1_routes[a2]
                            )
                        )
                routes[(a1, a2)] = a1_routes[a2]

    hosting_costs = {}
    default_cost = 0
    default_agt_costs = {}
    if "hosting_costs" in loaded:
        costs = loaded["hosting_costs"]
        for a in costs:
            if a == "default":
                default_cost = costs["default"]
                continue
            if a not in agents_list:
                raise DcopInvalidFormatError("hosting_costs for unknown " "agent " + a)
            a_costs = costs[a]
            if "default" in a_costs:
                default_agt_costs[a] = a_costs["default"]
            if "computations" in a_costs:
                for c in a_costs["computations"]:
                    hosting_costs[(a, c)] = a_costs["computations"][c]

    # Now that we parsed all agents info, we can build the objects:
    agents = {}
    for a in agents_list:
        d = default_cost
        if a in default_agt_costs:
            d = default_agt_costs[a]
        p = {c: hosting_costs[b, c] for (b, c) in hosting_costs if b == a}

        routes_a = {a2: v for (a1, a2), v in routes.items() if a1 == a}
        routes_a.update({a1: v for (a1, a2), v in routes.items() if a2 == a})

        agents[a] = AgentDef(
            a,
            default_hosting_cost=d,
            hosting_costs=p,
            default_route=default_route,
            routes=routes_a,
            **agents_list[a]
        )

    return agents


def yaml_agents(agents: List[AgentDef]) -> str:
    """
    Serialize a list of agents into a json string.

    Parameters
    ----------
    agents: list
        a list of agents

    Returns
    -------
    string:
        a json string representing the list of agents

    """
    agt_dict = {}
    hosting_costs = {}
    routes = {}
    for agt in agents:
        if hasattr(agt, "capacity"):
            agt_dict[agt.name] = {"capacity": agt.capacity}
        else:
            agt_dict[agt.name] = {}
        if agt.default_hosting_cost or agt.hosting_costs:
            hosting_costs[agt.name] = {
                "default": agt.default_hosting_cost,
                "computations": agt.hosting_costs,
            }
        if agt.routes:
            routes[agt.name] = agt.routes
        if agt.default_route is not None:
            routes["default"] = agt.default_route

    res = {}
    if agt_dict:
        res["agents"] = agt_dict
    if routes:
        res["routes"] = routes
    if hosting_costs:
        res["hosting_costs"] = hosting_costs

    if res:
        return yaml.dump(res, default_flow_style=False)
    else:
        return ""


def _build_dist_hints(loaded, dcop):
    if "distribution_hints" not in loaded:
        return None
    loaded = loaded["distribution_hints"]

    must_host, host_with = None, None
    if "must_host" in loaded:
        for a in loaded["must_host"]:
            if a not in dcop.agents:
                raise ValueError(
                    "Cannot use must_host with unknown agent " "{}".format(a)
                )
            for c in loaded["must_host"][a]:
                if c not in dcop.variables and c not in dcop.constraints:
                    raise ValueError(
                        "Cannot use must_host with unknown "
                        "variable or constraint {}".format(c)
                    )

        must_host = loaded["must_host"]

    if "host_with" in loaded:
        host_with = defaultdict(lambda: set())
        for i in loaded["host_with"]:
            host_with[i].update(loaded["host_with"][i])
            for j in loaded["host_with"][i]:
                s = {i}.union(loaded["host_with"][i])
                s.remove(j)
                host_with[j].update(s)

    return DistributionHints(
        must_host, dict(host_with) if host_with is not None else {}
    )


def str_2_domain_values(domain_str):
    """
    Deserialize a domain expressed as a string.

    If all variable in the domain can be interpreted as a int, the list is a
    list of int, otherwise it is a list of strings.

    :param domain_str: a string like 0..5 of A, B, C, D

    :return: the list of values in the domain
    """
    try:
        sep_index = domain_str.index("..")
        # Domain str is : [0..5]
        min_d = int(domain_str[0:sep_index])
        max_d = int(domain_str[sep_index + 2 :])
        return list(range(min_d, max_d + 1))
    except ValueError:
        values = [v.strip() for v in domain_str[1:].split(",")]
        try:
            return [int(v) for v in values]
        except ValueError:
            return values


def load_scenario_from_file(filename: str) -> Scenario:
    """
    Load a scenario from a yaml file.
    :param filename:
    :return:
    """
    with open(filename, mode="r", encoding="utf-8") as f:
        content = f.read()
    if content:
        return load_scenario(content)


def load_scenario(scenario_str) -> Scenario:
    """
    Load a scenario from a yaml string.
    :param scenario_str:
    :return:
    """
    loaded = yaml.load(scenario_str, Loader=yaml.FullLoader)
    evts = []
    for evt in loaded["events"]:
        id_evt = evt["id"]
        if "actions" in evt:
            actions = []
            for a in evt["actions"]:
                args = dict(a)
                args.pop("type")
                actions.append(EventAction(a["type"], **args))
            evts.append(DcopEvent(id_evt, actions=actions))
        elif "delay" in evt:
            evts.append(DcopEvent(id_evt, delay=evt["delay"]))

    return Scenario(evts)


def yaml_scenario(scenario: Scenario) -> str:
    events = [_dict_event(event) for event in scenario.events]
    scenario_dict = {"events": events}

    return yaml.dump(scenario_dict, default_flow_style=False)


def _dict_event(event: DcopEvent) -> Dict:
    evt_dict = {"id": event.id}
    if event.is_delay:
        evt_dict["delay"] = event.delay
    else:
        print(f" event {event}")
        evt_dict["actions"] = [_dict_action(a) for a in event.actions]
    return evt_dict


def _dict_action(action: EventAction) -> Dict:
    action_dict = {"type": action.type}
    action_dict.update(action.args)
    return action_dict
