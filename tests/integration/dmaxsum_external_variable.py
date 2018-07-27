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
Dynamic MaxSum with external variable.

This test/ sample is only meant to demonstrate the dynamic-maxsum algorithm :
deployment and distribution of the computation are done manually.
It is also on purpose completely independent of secp / smart-lighting code.

Graph-coloring problem with 3 colors and 4 nodes. Each node has one preferred
color.
Variables :
  * v1, prefers R
  * v2, prefers B
  * v3, prefers B
  * v4, prefers R
  * e1 : external variable, boolean

Constraints
  * R1:  3-ary constraint: v1 != v2, v2 != v3, v3 != v3 -
    ONLY active when e1 is true
  * R2:  v2 != v2
  * R3:  v3 != v4

When e1 is false :
  * R1 is inactive
  * Each variable can take it's favorite color, the best assignment is
    V1: R, V2: R, V3: B, v4: R

When e1 is True
  * R1 is active, so v2 and v3 cannot both take the value 'B'
  * V3 or v2 take 'G', all other variables take their favorite color

"""
import logging
import sys
import time

from pydcop.algorithms import filter_assignment_dict
from pydcop.infrastructure.computations import ExternalVariableComputation
from pydcop.algorithms.maxsum_dynamic import DynamicFactorComputation, \
    DynamicFactorVariableComputation
from pydcop.dcop.objects import VariableDomain, VariableNoisyCostFunc, \
    ExternalVariable
from pydcop.dcop.relations import NAryFunctionRelation, \
    find_dependent_relations, ConditionalRelation
from pydcop.infrastructure.run import Agent, AgentsRunner
from pydcop.infrastructure.communication import InProcessCommunicationLayer

logging.basicConfig(level=logging.DEBUG)
logging.info('Dynamic Maxsum with external variable')


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


def dmaxsum_external_variable():

    domain = VariableDomain('colors', 'color', ['R', 'G', 'B'])
    # RW Variables
    v1 = VariableNoisyCostFunc('v1', domain, prefer_color('R'))
    v2 = VariableNoisyCostFunc('v2', domain, prefer_color('B'))
    v3 = VariableNoisyCostFunc('v3', domain, prefer_color('B'))
    v4 = VariableNoisyCostFunc('v4', domain, prefer_color('R'))

    # External Variable
    domain_e = VariableDomain('boolean', 'abstract', [False, True])
    e1 = ExternalVariable('e1', domain_e, False)

    def r1(v1_, v2_, v3_):
        if v1_ != v2_ and v2_ != v3_ and v1_ != v3_:
            return 0
        return 100

    condition = NAryFunctionRelation(lambda x: x, [e1], name='r1_cond')
    relation_if_true = NAryFunctionRelation(r1, [v1, v2, v3], name='r1')
    r1 = ConditionalRelation(condition, relation_if_true)

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

    r1_computation = DynamicFactorComputation(r1, name='r1')
    r2_computation = DynamicFactorComputation(r2)
    r3_computation = DynamicFactorComputation(r3)

    e1_computation = ExternalVariableComputation(e1)

    # MUST only consider current relation when building computation objects !!
    # When a relation uses external variable, these must be sliced out.
    current_r1 = r1.slice({e1.name: e1.value})
    relations = [current_r1, r2, r3]
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

    a4 = Agent('a4', comm)
    a4.add_computation(e1_computation)

    agents = [a1, a2, a3, a4]
    runner = AgentsRunner(agents)
    runner.run_agents()

    # Now change a factor function every two seconds
    fail = False
    for i in range(5):
        time.sleep(2)
        current_value = e1_computation.current_value
        print('###  Iteration {} - function {}'.format(i, current_value))
        print(runner.status_string())
        results = runner.variable_values()
        if current_value:
            c = r1(filter_assignment_dict(results, r1.dimensions)) + \
                r2(filter_assignment_dict(results, r2.dimensions)) + \
                r3(filter_assignment_dict(results, r3.dimensions))
            if c != 0:
                print('Error on results for {} : \nGot {}  !'.format(
                        current_value, results))
                fail = True
                break
        else:
            c = r2(filter_assignment_dict(results, r2.dimensions)) + \
                r3(filter_assignment_dict(results, r3.dimensions))
        if c != 0:
            print('Error on results for {} : \nGot {} !'.format(
                current_value, results))
            fail = True
            break

        new_val = not current_value
        print('## Changing e1 value to {}'.format(new_val))
        e1_computation.change_value(new_val)

    print('Finished, stopping agents')
    runner.request_stop_agents(wait_for_stop=True)

    if fail:
        print('Failed !')
        return 1
    else:
        print('Success !')
        return 0


def run_test():
    return dmaxsum_external_variable()


if __name__ == "__main__":
    res = dmaxsum_external_variable()

    sys.exit(res)
