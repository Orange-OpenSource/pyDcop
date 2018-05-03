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

from pydcop.dcop import relations
from pydcop.algorithms.dpop import DpopAlgo
from pydcop.dcop.objects import Variable
from pydcop.infrastructure.run import synchronous_single_run
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer

# Sample using an unary constraint
#
# Constraint Network:
#
#  * Two variables x0 and x1
#  * Two constraints
#    * one unary constraint on x0: preference order a > c > b
#    * one binary constraint between x0 and x1: must be different


logging.basicConfig(level=logging.DEBUG)
logging.info('DPOP sample : unary constraint ')


def dpop_unary_constraint():
    x0 = Variable('x0', ['a', 'b', 'c'])
    x1 = Variable('x1', ['a', 'b', 'c'])

    # Unary constraint on x0
    @relations.AsNAryFunctionRelation(x0)
    def unary_x1(x):
        if x == 'a':
            return 8
        if x == 'b':
            return 2
        if x == 'c':
            return 5

    @relations.AsNAryFunctionRelation(x0, x1)
    def prefer_different(x, y):
        if x == y:
            return 0
        else:
            return 10

    al0 = DpopAlgo(x0)
    al1 = DpopAlgo(x1)

    # Distribution: two agents
    comm = InProcessCommunicationLayer()
    a0 = Agent('a0', comm)
    a0.add_computation(al0)
    a1 = Agent('a1', comm)
    a1.add_computation(al1)

    al0.set_parent(x1)
    al0.add_relation(prefer_different)
    al1.add_child(x0)

    # we represent the unary constraint as a variable having itself as
    # pseudo-parent
    al0.add_relation(unary_x1)

    results, _, _ = synchronous_single_run([a0, a1])

    if results == {'x0': 'a', 'x1': 'b'} or \
       results == {'x0': 'a', 'x1': 'c'}:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return dpop_unary_constraint()

if __name__ == "__main__":
    sys.exit(dpop_unary_constraint())
