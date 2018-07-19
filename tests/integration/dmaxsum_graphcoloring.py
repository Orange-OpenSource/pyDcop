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


import sys
import time

from pydcop.algorithms.maxsum_dynamic import DynamicFactorComputation, \
    DynamicFactorVariableComputation
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import VariableNoisyCostFunc
from pydcop.dcop.relations import find_dependent_relations, \
    NAryFunctionRelation
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer

"""
Graph coloring sample with dynamic Maxsum
4 variable that can take one of 3 colors

One relation (r1) is periodically changed (with dimension change).

In this sample we do not use read-only variables but we directly change the
function on the factor node.

"""

# Domain for colors
d = ['R', 'G', 'B']


def prefer_color(preferred_color):
    """ Generates a cost function for a variable that prefers one specific
    color
    """
    def color_cost(color):
        if color == preferred_color:
            return 0
        else:
            return 5
    return color_cost


def dmaxsum_graphcoloring():

    v1 = VariableNoisyCostFunc('v1', d, prefer_color('R'))
    v2 = VariableNoisyCostFunc('v2', d, prefer_color('G'))
    v3 = VariableNoisyCostFunc('v3', d, prefer_color('B'))
    v4 = VariableNoisyCostFunc('v4', d, prefer_color('R'))

    def r1(v1_, v2_, v3_):
        if v1_ != v2_ and v2_ != v3_ and v1_ != v3_:
            return 0
        return 100

    r1 = NAryFunctionRelation(r1, [v1, v2, v3], name='r1')

    def r1_2(v1_, v2_, v4_):
        if v1_ != v2_ and v2_ != v4_ and v1_ != v4_:
            return 0
        return 100

    r1_2 = NAryFunctionRelation(r1_2, [v1, v2, v4], name='r1_2')

    def r2(v2_, v4_):
        if v2_ != v4_:
            return 0
        return 100

    r2 = NAryFunctionRelation(r2, [v2, v4], name='r2')

    def r3(v3_, v4_):
        if v3_ != v4_:
            return 0
        return 100

    r3 = NAryFunctionRelation(r3, [v3, v4], name='r3')

    relations = [r1, r2, r3]
    r1_computation = DynamicFactorComputation(r1, name='r1')
    r2_computation = DynamicFactorComputation(r2)
    r3_computation = DynamicFactorComputation(r3)

    v1_computation = \
        DynamicFactorVariableComputation(
            v1,
            [r.name for r in find_dependent_relations(v1, relations)])
    v2_computation = \
        DynamicFactorVariableComputation(
            v2,
            [r.name for r in find_dependent_relations(v2, relations)])
    v3_computation = \
        DynamicFactorVariableComputation(
            v3,
            [r.name for r in find_dependent_relations(v3, relations)])
    v4_computation = \
        DynamicFactorVariableComputation(
            v4,
            [r.name for r in find_dependent_relations(v4, relations)])

    # Prepare the agents
    comm = InProcessCommunicationLayer()
    a1 = Agent('a1', comm)
    a1.add_computation(v1_computation)
    a1.add_computation(r1_computation)

    a2 = Agent('a2', comm)
    a2.add_computation(v2_computation)
    a1.add_computation(r2_computation)

    a3 = Agent('a3', comm)
    a3.add_computation(v3_computation)
    a3.add_computation(v4_computation)
    a3.add_computation(r3_computation)

    # Expected results to check for success
    expected_results = {'r1': {'v1': 'R', 'v2': 'G', 'v3': 'B', 'v4': 'R'},
                        'r1_2': {'v1': 'B', 'v2': 'G', 'v3': 'B', 'v4': 'R'}}
    r1_fcts = [r1, r1_2]

    agents = [a1, a2, a3]

    for a in agents:
        a.start()

    # Now change a factor function every two seconds
    r1_fct, iteration, fail = 0, 0, False
    for _ in range(5):
        iteration += 1
        time.sleep(2)
        print('###  Iteration {} - function {}'.format(iteration, r1_fcts[
              r1_fct].name))
        print(runner.status_string())
        results = runner.variable_values()
        if not results == expected_results[r1_fcts[r1_fct].name]:
            print('Error on results for {} : \nGot {} instead of {}  !'.format(
                r1_fcts[r1_fct].name, results, expected_results[r1_fcts[
                    r1_fct].name]
            ))
            fail = True
            break
        r1_fct = (r1_fct + 1) % 2
        print('## Changing to function {}'.format(r1_fcts[r1_fct].name))
        r1_computation.change_factor_function(r1_fcts[r1_fct])

    print('Finished, stopping agents')
    runner.request_stop_agents(wait_for_stop=True)

    if fail:
        print('Failed !')
        return 1
    else:
        print('Success !')
        return 0


def run_test():
    return dmaxsum_graphcoloring()


if __name__ == "__main__":
    res = dmaxsum_graphcoloring()

    sys.exit(res)
