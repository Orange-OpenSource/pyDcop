
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


def generate_secp(args):
    logger.info("Generate SECP %s", args)
    light_count = args.lights
    model_count = args.models
    rule_count = args.rules
    capacity = args.capacity
    max_model_size = 3

    light_domain = Domain("light", "light", range(0, 5))

    lights_var, lights_cost = build_lights(light_count, light_domain)

    models_var, models_constraints = build_models(
        light_domain, lights_var, max_model_size, model_count
    )

    rules_constraints = build_rules(rule_count, lights_var, models_var)

    # Agents : one for each light
    agents = build_agents(lights_var, capacity)

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


def build_agents(lights_var, capacity=None):
    agents = {}
    for light_var in lights_var:
        hosting_costs = {light_var: 0}
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
        model_expression = "0 if {} == {} else 10000 ".format(
            light_expression, model_var.name
        )
        model = constraint_from_str(
            "c_m{}".format(j),
            expression=model_expression,
            all_variables=list(lights.values()) + [model_var],
        )
        models[model.name] = model

    return models_var, models


def build_rules(rule_count, lights_var, models_var):
    # Rules : one constraint with a target for a model or a light
    # Example:
    #   function: 10 * (abs(mv_livingroom - 5) + abs(mv_kitchen - 4))
    all_variables = list(lights_var.values()) + list(models_var.values())
    rules_constraints = {}
    for k in range(rule_count):

        if random() < 0.7:
            # model based rule
            max_size = min(4, len(models_var))
            rule_size = randint(1, max_size)
            rules_models = sample(list(models_var), rule_size)
            model_expression_parts = []
            for model_var in rules_models:
                target = randint(0, 9)
                model_expression_part = "abs({} - {} )".format(model_var, target)
                model_expression_parts.append(model_expression_part)

            model_expression = " + ".join(model_expression_parts)
            rule_expression = "10 * ( {} )".format(model_expression)
            rule = constraint_from_str(
                "r_{}".format(k),
                expression=rule_expression,
                all_variables=all_variables,
            )
        else:
            # light based rule
            target = randint(0, 9)
            light = choice(list(lights_var))
            rule_expression = "10 * abs({} - {})".format(light, target)
            rule = constraint_from_str(
                "r_{}".format(k),
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
