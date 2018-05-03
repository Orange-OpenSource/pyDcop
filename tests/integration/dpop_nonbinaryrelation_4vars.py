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
logging.info('DPOP tests with non-binary relations & 4 variables')


# * 3 variables
# * one 3-ary relation between these variable
# * use minimization


# * One agent for each variable
INFNT = sys.maxsize


def dpop_nonbinaryrelation_4vars():

    x0 = Variable('x0', list(range(10)))
    x1 = Variable('x1', list(range(10)))
    x2 = Variable('x2', list(range(10)))
    x3 = Variable('x3', list(range(10)))

    @relations.AsNAryFunctionRelation(x0)
    def x0_prefs(x):
        if x > 3:
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x1)
    def x1_prefs(x):
        if 2 < x < 7:
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x2)
    def x2_prefs(x):
        if x < 5:
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x3)
    def x3_prefs(x):
        if 0 < x < 5:
            return 0
        return 10

    @relations.AsNAryFunctionRelation(x0, x1, x2, x3)
    def four_ary_relation(a, b, c, d):
        return abs(10 - (a+b+c+d))

    def neutral_relation(x, y):
        return 0

    comm = InProcessCommunicationLayer()

    al0 = DpopAlgo(x0, mode='min')
    al1 = DpopAlgo(x1, mode='min')
    al2 = DpopAlgo(x2, mode='min')
    al3 = DpopAlgo(x3, mode='min')

    # unary relation for preferences
    al0.add_relation(x0_prefs)
    al1.add_relation(x1_prefs)
    al2.add_relation(x2_prefs)
    al3.add_relation(x3_prefs)

    # The 4-ary relation must be introduced only once, in the lowest node in
    # the DFS tree (in our case, a3).
    # We still need to add relation between the
    # We use neutral relation between two variables that are present in the
    # tree but that do not share a real constraint : in reality with the
    # current implementation, these neutral relation are not needed any-more,
    #  I just keep them here for historical reasons.

    al0.add_child(x1)
    al1.set_parent(x0)
    al1.add_relation(relations.NAryFunctionRelation(
        neutral_relation, [x0, x1]))

    al1.add_child(x2)
    al2.set_parent(x1)
    al2.add_relation(relations.NAryFunctionRelation(
        neutral_relation, [x1, x2]))

    al2.add_child(x3)
    al3.set_parent(x2)
    al3.add_relation(relations.NAryFunctionRelation(
        neutral_relation, [x2, x3]))

    al3.add_relation(four_ary_relation)

    a0 = Agent('a0', comm)
    a0.add_computation(al0)
    a1 = Agent('a1', comm)
    a1.add_computation(al1)
    a2 = Agent('a2', comm)
    a2.add_computation(al2)
    a3 = Agent('a3', comm)
    a3.add_computation(al3)

    results, _, _ = synchronous_single_run([a0, a1, a2, a3])

    if results == {'x0': 4, 'x2': 0, 'x1': 3, 'x3': 3}:
        logging.info('SUCCESS !! ' + str(results))
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return dpop_nonbinaryrelation_4vars()


if __name__ == "__main__":
    sys.exit(dpop_nonbinaryrelation_4vars())
