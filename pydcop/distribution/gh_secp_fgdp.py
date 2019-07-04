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

GH-SECP-FGDP : Greedy Heuristic for the SECP Factor Graph Distribution Problem

To minimize the communication load, we place each pair (yj , φj) on an agent ai with i chosen
such that xi ∈ Sφj , meaning that xi is one of the variables influencing yj . Similarly, a factor rk
is hosted on an agent ai such that xi ∈ Srk . Intuitively this means that the factor representing a
rule is always hosted on a agent affected by this rule. As to ensure a balanced computation load,
yj ’s, φj ’s and rk ’s are fairly distributed among the candidate agents.

When implementing this heuristic, we use a greedy approach to select
one agent among the set of valid agents for a given computation: we select the agent, with enough
capacity, that is already hosting the highest number of computations that share a dependency with
the computation we are placing. In case of tie, we chose the agent with the highest remaining
capacity. By grouping interdependent computations, this approach favors distributions with a low
communication cost.

* actuator variable and cost factor are placed on the corresponding device/ agent,
  which is the agent where the hosting cost is 0
* variable and factor for physicals model are hosted on the same agent.
  We select that agent by looking at the model's factor and look for the
  the agent with enough capacity that already host the highest number of neighbors

Notice that
* the communication load is not used in this distribution method
* the computation footprint is required

This distribution method is designed for SECP and makes the following assumptions on
the DCOP (which are always satisfied if the SECP has been generated using
'pydcop generate' command):
* variables that represent an actuator are assigned to an agent, with an hosting cost of 0
* If external cost constraints are used for actuators, they must be named after the
  actuator's name. E.g. (l0, c_l0)
* Physical model factor are named after the physical model variable name,
  e.g. (m1, c_m1)


"""
import logging
from typing import Iterable, Callable
from collections import defaultdict

from pydcop.computations_graph.factor_graph import (
    ComputationsFactorGraph,
    VariableComputationNode,
    FactorComputationNode,
)
from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.objects import AgentDef
from pydcop.distribution import oilp_secp_fgdp
from pydcop.distribution.objects import (
    DistributionHints,
    Distribution,
    ImpossibleDistributionException,
)
from pydcop.distribution.gh_secp_cgdp import find_candidates

logger = logging.getLogger("distribution.gh_secp_fgdp")


def distribute(
    computation_graph: ComputationsFactorGraph,
    agentsdef: Iterable[AgentDef],
    hints: DistributionHints = None,
    computation_memory: Callable[[ComputationNode], float] = None,
    communication_load: Callable[[ComputationNode, str], float] = None,
    timeout=None,  # not used
) -> Distribution:
    if computation_memory is None:
        raise ImpossibleDistributionException(
            "adhoc distribution requires " "computation_memory functions"
        )

    # as we're dealing with a secp modelled as a factor graph, we have computations for
    # actuator and physical model variables, rules and physical model factors.
    mapping = defaultdict(lambda: [])
    agents_capa = {a.name: a.capacity for a in agentsdef}
    variable_computations = []
    factor_computations = []
    for comp in computation_graph.nodes:
        if isinstance(comp, VariableComputationNode):
            variable_computations.append(comp.name)
        elif isinstance(comp, FactorComputationNode):
            factor_computations.append(comp.name)
        else:
            raise ImpossibleDistributionException(
                f"Error: {comp} is neither a factor nor a variable computation"
            )

    # First, put each actuator variable and cost factor on its agent
    for variable in variable_computations[:]:

        for agent in agentsdef:
            if agent.hosting_cost(variable) == 0:
                # Found an actuator variable, host it on the agent
                mapping[agent.name].append(variable)
                variable_computations.remove(variable)
                agents_capa[agent.name] -= computation_memory(
                    computation_graph.computation(variable)
                )
                # search for the cost factor, if any, and host it on the same agent.
                for factor in factor_computations[:]:
                    if f"c_{variable}" == factor:
                        mapping[agent.name].append(factor)
                        factor_computations.remove(factor)
                        agents_capa[agent.name] -= computation_memory(
                            computation_graph.computation(factor)
                        )
                if agents_capa[agent.name] < 0:
                    raise ImpossibleDistributionException(
                        f"Not enough capacity on {agent} to hosts actuator {variable}: {agents_capa[agent.name]}"
                    )
                break
    logger.info(f"Actuator computations - agents: {dict(mapping)}")
    logger.info(f"Remaining capacity: {dict(agents_capa)}")

    # now find computations for physical models and variables variables.
    # * all remaining variables are model variables
    # * physical model factor computation names contain the name of the variable
    model_variables = variable_computations
    models = []
    for model_var in model_variables:
        for fact in factor_computations:
            if f"c_{model_var}" == fact:
                models.append((model_var, fact))
                factor_computations.remove(fact)

    # All remaining factor ar rule factors
    rule_factors = factor_computations

    logger.debug(f"Physical models: {models}")
    logger.debug(f"Rules: {rule_factors}")

    # Now place models
    for model_var, model_fac in models:
        footprint = computation_memory(
            computation_graph.computation(model_fac)
        ) + computation_memory(computation_graph.computation(model_var))
        neighbors = computation_graph.neighbors(model_fac)

        candidates = find_candidates(
            agents_capa, model_fac, footprint, mapping, neighbors
        )

        # Host the model on the first agent and decrease its remaining capacity
        selected = candidates[0][2]
        mapping[selected].append(model_var)
        mapping[selected].append(model_fac)
        agents_capa[selected] -= footprint
    logger.debug(f"All models hosted: {dict(mapping)}")
    logger.debug(f"Remaining capacity: {agents_capa}")

    # And rules at last:
    for rule_fac in rule_factors:
        footprint = computation_memory(computation_graph.computation(rule_fac))
        neighbors = computation_graph.neighbors(rule_fac)

        candidates = find_candidates(
            agents_capa, rule_fac, footprint, mapping, neighbors
        )

        # Host the computation on the first agent and decrease its remaining capacity
        selected = candidates[0][2]
        mapping[selected].append(rule_fac)
        agents_capa[selected] -= footprint

    return Distribution({a: list(mapping[a]) for a in mapping})


def distribution_cost(
    distribution: Distribution,
    computation_graph: ComputationGraph,
    agentsdef: Iterable[AgentDef],
    computation_memory: Callable[[ComputationNode], float],
    communication_load: Callable[[ComputationNode, str], float],
) -> float:
    return oilp_secp_fgdp.distribution_cost(
        distribution,
        computation_graph,
        agentsdef,
        computation_memory,
        communication_load,
    )
