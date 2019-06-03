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


import logging
import sys

import pydcop.infrastructure.communication
from pydcop import infrastructure
from pydcop.algorithms import amaxsum
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable

INFNT = sys.maxsize


logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum Smart Lighting test 1')


def maxsum_smartlights_simple():
    """
    First SCEP implementation :

     * 3 lights l1, l2 & l3
       each light can have a luminosity level in the  0-9 range
       The energy cost is a linear function of the luminosity level, with l1
       more efficient than l2 and l3

     * one scene action y1, the room luminosity level
       y1 = (l1 + l2 + l3 )/3
       y1 domain is also between 0-9

     * one rule :
       l3 must be off AND y1 Must be 5

    No stop condition is implemented yet, the algorithm must be stooped
    manually.

    """
    # building the Factor Graph
    # Lights :

    l1 = Variable('l1', list(range(10)))

    @relations.AsNAryFunctionRelation(l1)
    def cost_l1(l1):
        return 0.5 * l1

    l2 = Variable('l2', list(range(10)))

    @relations.AsNAryFunctionRelation(l2)
    def cost_l2(l2):
        return l2

    l3 = Variable('l3', list(range(10)))

    @relations.AsNAryFunctionRelation(l3)
    def cost_l3(l3):
        return l3

    # Scene action
    y1 = Variable('y1', list(range(10)))

    @relations.AsNAryFunctionRelation(l1, l2, l3, y1)
    def scene_rel(l1, l2, l3, y1):
        if y1 == round(l1 / 3.0 + l2 / 3.0 + l3 / 3.0):
            return 0
        return INFNT

    # Rule
    @relations.AsNAryFunctionRelation(l3, y1)
    def rule_rel(l3, y1):
        """
        This rule means : target luminosity if 5, and x3 is off.

        :param x3:
        :param y1:
        :return:
        """
        return 10 * (abs(y1 - 5) + l3)

    # Create computation for factors and variables
    # Light 1
    algo_l1 = amaxsum.VariableAlgo(l1, [cost_l1.name, scene_rel.name])
    algo_cost_l1 = amaxsum.FactorAlgo(cost_l1)

    # Light 2
    algo_l2 = amaxsum.VariableAlgo(l2, [cost_l2.name, scene_rel.name])
    algo_cost_l2 = amaxsum.FactorAlgo(cost_l2)

    # Light 3
    algo_l3 = amaxsum.VariableAlgo(l3, [cost_l3.name, scene_rel.name,
                                        rule_rel.name])
    algo_cost_l3 = amaxsum.FactorAlgo(cost_l3)

    # Scene
    algo_y1 = amaxsum.VariableAlgo(y1, [rule_rel.name, scene_rel.name])
    algo_scene = amaxsum.FactorAlgo(scene_rel)

    #Rule
    algo_rule = amaxsum.FactorAlgo(rule_rel)

    # Distribution of the computation on the three physical light-bulb nodes.
    # We have 9 computations to distribute on 3 agents, mapping the 3 light
    # bulbs.
    comm = infrastructure.communication.InProcessCommunicationLayer()

    a1 = infrastructure.Agent('Bulb1', comm)
    a1.add_computation(algo_cost_l1)
    a1.add_computation(algo_l1)
    a1.add_computation(algo_scene)
    a1.add_computation(algo_y1)

    a2 = infrastructure.Agent('Bulb2', comm)
    a2.add_computation(algo_cost_l2)
    a2.add_computation(algo_l2)

    a3 = infrastructure.Agent('Bulb3', comm)
    a3.add_computation(algo_cost_l3)
    a3.add_computation(algo_l3)
    a3.add_computation(algo_rule)

    dcop_agents = [a1, a2, a3]

    results, _, _ = infrastructure.synchronous_single_run(dcop_agents)

    print(results)

    if results == {'l1': 9, 'y1': 5, 'l3': 0, 'l2': 5}:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return maxsum_smartlights_simple()


if __name__ == "__main__":
    sys.exit(maxsum_smartlights_simple())