# BSD-3-Clause License
#
# Copyright 2018 Orange
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
.. _pydcop_commands_generate_scep:


pydcop generate secp
====================

Generate a SECP.

Synopsis
--------
::

    pydcop generate secp
                      --lights <lights_counts>
                      --models <models_counts>
                      --rules <rules_count>
                      --capacity <capacity>
                      [--max_model_size <max_model_size>]
                      [--max_rule_size <max_rule_size>]

Description
-----------

Generate a DCOP representing a SECP.

* Each light has a random efficiency factor used to generate it's cost function.
* Each model involves at random from at most `max_model_size` lights.
* Each rule involves at random for at most `max_rule_size` models and lights and
  sets a random target value for each of these elements.

Options
-------

``--lights <lights_counts>``
  Number of lights in the SECP.

``--models <models_counts>``
  Number of models in the SECP.

``--rules <rules_counts>``
  Number of rules in the SECP.

``--capacity <capacity>``
  capacity of an agent

``--max_model_size <max_model_size>``
  The maximum number of lights involved in a model.
  Defaults to 3.

``--max_rule_size <max_rule_size>``
  The maximum number of elements (lights and models) involved in a rule.
  Defaults to 3.


Examples
========

Generating a DCOP for a SECP with 10 lights, 3 models and 2 rules.::

  pydcop generate secp --lights 10 --models 3 --rules 2

"""
import logging
import os
from random import randint, sample, choice, random

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, Variable, AgentDef
from pydcop.dcop.relations import constraint_from_str
from pydcop.dcop.yamldcop import dcop_yaml


logger = logging.getLogger("pydcop.generate")


def parser_secp(subparser):
    parser = subparser.add_parser("secp", help="generate an secp")
    parser.set_defaults(func=generate_secp)
    parser.add_argument(
        "-l", "--lights", type=int, required=True, help="number of lights"
    )
    parser.add_argument(
        "-m", "--models", type=int, required=True, help="number of models"
    )
    parser.add_argument(
        "-r", "--rules", type=int, required=True, help="number of rules"
    )
    parser.add_argument(
        "-c", "--capacity", type=int, default=None, help="agent's capacity"
    )
    parser.add_argument(
        "--max_model_size", type=int, default=3, help="maximum number of lights involved in a model"
    )
    parser.add_argument(
        "--max_rule_size", type=int, default=3, help="maximum number of elements involved in a rule"
    )


def generate_secp(args):
    logger.info("Generate SECP %s", args)
    light_count = args.lights
    model_count = args.models
    rule_count = args.rules
    capacity = args.capacity
    max_model_size = args.max_model_size
    max_rule_size = args.max_rule_size

    light_domain = Domain("light", "light", range(0, 5))

    lights_var, lights_cost = build_lights(light_count, light_domain)

    models_var, models_constraints = build_models(
        light_domain, lights_var, max_model_size, model_count
    )

    rules_constraints = build_rules(rule_count, lights_var, models_var, max_rule_size)

    # Agents : one for each light
    agents = build_agents(lights_var, lights_cost, capacity)

    # Force
    # * each light variable to be hosted on the corresponding agent
    # * model constraint and var are preferred on the same agent

    variables = lights_var.copy()
    variables.update(models_var)

    constraints = models_constraints.copy()
    constraints.update(lights_cost)
    constraints.update(rules_constraints)

    dcop = DCOP(
        "graph coloring",
        "min",
        domains={"light_domain": light_domain},
        variables=variables,
        agents=agents,
        constraints=constraints,
    )

    if args.output:
        outputfile = args.output
        write_in_file(outputfile, dcop_yaml(dcop))
    else:
        print(dcop_yaml(dcop))


def build_agents(lights_vars, lights_costs, capacity=None):
    agents = {}
    for light_var, light_cost in zip(lights_vars, lights_costs):
        hosting_costs = {light_var: 0, light_cost: 0}
        logger.debug(f"Creating agent for {light_var} with hosting {hosting_costs}")
        if capacity:
            agt = AgentDef(
                "a{}".format(light_var),
                hosting_costs=hosting_costs,
                capacity=capacity,
                default_hosting_cost=100,
            )
        else:
            agt = AgentDef(
                "a{}".format(light_var),
                hosting_costs=hosting_costs,
                default_hosting_cost=100,
            )

        agents[agt.name] = agt
    return agents


def build_models(light_domain, lights, max_model_size, model_count):
    # Models: for each model
    #  * one constraint depends on 2 =< k =<max_model_size lights
    #  * one variable ,
    # Example:
    #   function: 0 if 0.7 * l_d1 + 0.5 * l_d2 + 0.3 * l_lv3 == mv_desk else
    #             1000
    # function: '0 if 10 * abs(m0 - ( 0.2 * l1 + 0.5 * l2 + 0.8 * l3 )) < 3 else 1000'

    models = {}
    models_var = {}
    for j in range(model_count):
        model_var = Variable("m{}".format(j), domain=light_domain)
        models_var[model_var.name] = model_var

        model_size = randint(2, max_model_size)
        model_lights = ()
        light_expression_parts = []
        for k, model_light in enumerate(sample(list(lights), model_size)):
            impact = randint(1, 7) / 10
            light_expression_parts.append(" {} * {}".format(model_light, impact))
            # model_lights.append((model_light, impact))
            # model_light.
        light_expression = " + ".join(light_expression_parts)
        model_expression = f"0 if 10* abs({model_var.name} - ({light_expression})) < 5 else 10000 ".format(
            light_expression, model_var.name
        )
        model = constraint_from_str(
            "c_m{}".format(j),
            expression=model_expression,
            all_variables=list(lights.values()) + [model_var],
        )
        models[model.name] = model

    return models_var, models


def build_rules(rule_count, lights_var, models_var, max_rule_size):
    """
    Build a set of rules for the given lights and models.

    A rule set a target for some lights and or models.

    Rule depends at random for at most `max_rule_size` models and lights.

    Parameters
    ----------
    rule_count: int
        number of rule to generate
    lights_var: list of string
        names of light variables
    models_var: list of string
        names of model variables
    max_rule_size: int
        maximum number of element involved in a rule.

    Returns
    -------
    A dict containing the rules, indexed by their name.
    """
    # Rules : one constraint with a target for a model or a light
    # Example:
    #   function: 10 * (abs(mv_livingroom - 5) + abs(mv_kitchen - 4))
    all_variables = list(lights_var.values()) + list(models_var.values())
    rules_constraints = {}
    for k in range(rule_count):
        # set rule size
        max_size = min(max_rule_size, len(models_var) + len(lights_var))
        rule_size = randint(1, max_size)

        # A rule sets targets for lights and models.
        # it is represented by a function that returns a distance to these targets.
        # "10 * (abs(l3 - 4) + abs(m0 - 6 ) + abs(m2 - 4 ) )"

        # Lights in the rule
        # Example: "abs(l3 - 4)"
        lights_count = randint(0, rule_size)
        rules_lights = sample(list(lights_var), lights_count)
        expression_parts = []
        for light_var in rules_lights:
            target = randint(0, 4)
            light_expression_part = f"abs({light_var} - {target} )"
            expression_parts.append(light_expression_part)

        # Models in the rule
        # Example:  "abs(m0 - 6 ) + abs(m2 - 4 )"
        models_count = rule_size - lights_count
        rules_models = sample(list(models_var), models_count)
        for model_var in rules_models:
            target = randint(0, 4)
            model_expression_part = f"abs({model_var} - {target} )"
            expression_parts.append(model_expression_part)

        expression = " + ".join(expression_parts)
        rule_expression = f"10 * ({expression})"
        rule = constraint_from_str(
            f"r_{k}",
            expression=rule_expression,
            all_variables=all_variables,
        )
        rules_constraints[rule.name] = rule

    return rules_constraints


def build_lights(light_count, light_domain):
    # Lights : cost function & variable
    lights = {}
    lights_cost = {}
    for i in range(light_count):
        light = Variable("l{}".format(i), domain=light_domain)
        lights[light.name] = light
        efficiency = randint(0, 90) / 100
        cost = constraint_from_str(
            "c_l{}".format(i),
            expression="{} * {}".format(light.name, efficiency),
            all_variables=[light],
        )
        lights_cost[cost.name] = cost

    return lights, lights_cost


def write_in_file(filename: str, dcop_str: str):
    path = "/".join(filename.split("/")[:-1])

    if (path != "") and (not os.path.exists(path)):
        os.makedirs(path)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(dcop_str)
