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
from pydcop.algorithms.dpop import DpopAlgo
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable
from pydcop.infrastructure.run import Agent

logging.basicConfig(level=logging.DEBUG)
logging.info('DPOP sample 1 : 4 variables & 3 relations')

# This sample use the example described in Petcu Phd Thesis p56.
# The DCOP as 4 agents and three relations :
#
#    X0
#    |
#    X1
#   /  \
#  X2  x3
#
# Once finished the coorect solution is produced and can be read in the logs :
#   INFO:root:Agent a0 selects value a
#   INFO:root:Agent a1 selects value c
#   INFO:root:Agent a2 selects value b
#   INFO:root:Agent a3 selects value a


def dpop_petcu():
    comm = infrastructure.communication.InProcessCommunicationLayer()

    # Variables definition:

    x0 = Variable('x0', ['a', 'b', 'c'])
    x1 = Variable('x1', ['a', 'b', 'c'])
    x2 = Variable('x2', ['a', 'b', 'c'])
    x3 = Variable('x3', ['a', 'b', 'c'])

    # relation between variables

    r1_0 = relations.NAryMatrixRelation([x1, x0], [[2, 2, 3],
                                                   [5, 3, 7],
                                                   [6, 3, 1]])
    r2_1 = relations.NAryMatrixRelation([x2, x1], [[0, 2, 1],
                                                   [3, 4, 6],
                                                   [5, 2, 5]])
    r3_1 = relations.NAryMatrixRelation([x3, x1], [[6, 2, 3],
                                                   [3, 3, 2],
                                                   [4, 4, 1]])

    al0 = DpopAlgo(x0)
    al1 = DpopAlgo(x1)
    al2 = DpopAlgo(x2)
    al3 = DpopAlgo(x3)

    al0.add_child(x1)
    al1.add_child(x2)
    al1.add_child(x3)

    al2.set_parent(x1)
    al2.add_relation(r2_1)
    al3.set_parent(x1)
    al3.add_relation(r3_1)
    al1.set_parent(x0)
    al1.add_relation(r1_0)

    a0 = Agent('a0', comm)
    a0.add_computation(al0)
    a1 = Agent('a1', comm)
    a1.add_computation(al1)
    a2 = Agent('a2', comm)
    a2.add_computation(al2)
    a3 = Agent('a3', comm)
    a3.add_computation(al3)

    results, _, _ = infrastructure.synchronous_single_run([a0, a1, a2, a3])

    if results == {'x0': 'a', 'x1': 'c', 'x2': 'b', 'x3': 'a'}:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return dpop_petcu()

if __name__ == "__main__":
    sys.exit(dpop_petcu())
