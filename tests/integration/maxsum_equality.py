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

from pydcop.algorithms.amaxsum import VariableAlgo, FactorAlgo
from pydcop.dcop.objects import VariableWithCostFunc, VariableNoisyCostFunc
from pydcop.dcop.relations import AsNAryFunctionRelation
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.run import synchronous_single_run
from pydcop.infrastructure.communication import InProcessCommunicationLayer

logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum test With generator utility function')


def maxsum_equality_noise():
    """
    This sample demonstrates the use of noise to break ties between variables.
    """

    l1 = VariableNoisyCostFunc('l1', list(range(10)), lambda x: x)
    l2 = VariableNoisyCostFunc('l2', list(range(10)), lambda x: x)

    # Scene action
    y1 = VariableWithCostFunc('y1', list(range(10)), lambda x: 10 * abs(5-x))

    @AsNAryFunctionRelation(l1, l2, y1)
    def scene_rel(x, y, z):
        if z == x + y:
            return 0
        return 10000

    # Create computation for factors and variables
    # Light 1
    algo_l1 = VariableAlgo(l1, [scene_rel.name])

    # Light 2
    algo_l2 = VariableAlgo(l2, [scene_rel.name])

    # Scene
    algo_y1 = VariableAlgo(y1, [scene_rel.name])
    algo_scene = FactorAlgo(scene_rel)

    comm = InProcessCommunicationLayer()

    a1 = Agent('A1', comm)
    a1.add_computation(algo_l1)

    a2 = Agent('A2', comm)
    a2.add_computation(algo_l2)

    a3 = Agent('A3', comm)
    a3.add_computation(algo_y1)
    a3.add_computation(algo_scene)
    dcop_agents = [a1, a2, a3]

    results, _, _ = synchronous_single_run(dcop_agents, 5)

    print(results)

    if results['y1'] == 5 and results['l1'] + results['l2'] == 5:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return maxsum_equality_noise()


if __name__ == "__main__":
    sys.exit(maxsum_equality_noise())
