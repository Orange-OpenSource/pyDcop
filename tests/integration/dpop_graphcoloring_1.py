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

from pydcop.algorithms.dpop import DpopAlgo
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable
from pydcop.infrastructure.run import synchronous_single_run
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer

logging.basicConfig(level=logging.DEBUG)
logging.info('DPOP sample : Graph Coloring 1')

# Graph coloring problem:
#
#  * Use minimization
#  * 3 variables, all connected
#  * each variable has a preference for it's color, modelled as a unary
# relation


def dpop_graphcoloring_1():
    x0 = Variable('x0', ['R', 'G', 'B'])
    x1 = Variable('x1', ['R', 'G', 'B'])
    x2 = Variable('x2', ['R', 'G', 'B'])

    # Unary constraint on x0
    @relations.AsNAryFunctionRelation(x0)
    def x0_prefers_r(x):
        if x == 'R':
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x1)
    def x1_prefers_g(x):
        if x == 'G':
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x2)
    def x2_prefers_b(x):
        if x == 'B':
            return 0
        return 10

    def prefer_different(x, y):
        if x == y:
            return 10
        else:
            return 0

    r1_0 = relations.NAryFunctionRelation(prefer_different, [x0, x1])
    r0_2 = relations.NAryFunctionRelation(prefer_different, [x0, x2])
    r1_2 = relations.NAryFunctionRelation(prefer_different, [x1, x2])

    # Create computations objects to solve this problem
    # For this we must define the DFS tree
    # x0 ---> x1 ---> x2
    # preferences are modeled as unary relation, set directly on each variable
    # other constraints are represented by binary relations which are set on
    # the lowest variable in the tree.

    c0 = DpopAlgo(x0, mode='min')
    c0.add_child(x1)
    c0.add_relation(x0_prefers_r)

    c1 = DpopAlgo(x1, mode='min')
    c1.set_parent(x0)
    c1.add_child(x2)
    c1.add_relation(x1_prefers_g)
    c1.add_relation(r1_0)

    c2 = DpopAlgo(x2, mode='min')
    c2.set_parent(x1)
    c2.add_relation(x2_prefers_b)
    c2.add_relation(r0_2)
    c2.add_relation(r1_2)

    # Distribution: 3 agents, one for each variable
    comm = InProcessCommunicationLayer()
    a0 = Agent('a0', comm)
    a1 = Agent('a1', comm)
    a2 = Agent('a2', comm)

    a0.add_computation(c0)
    a1.add_computation(c1)
    a2.add_computation(c2)

    results, _, _ = synchronous_single_run([a0, a1, a2])

    if results == {'x0': 'R', 'x1': 'G', 'x2': 'B'}:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return dpop_graphcoloring_1()

if __name__ == "__main__":
    sys.exit(dpop_graphcoloring_1())
