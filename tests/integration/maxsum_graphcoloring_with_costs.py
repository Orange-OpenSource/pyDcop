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

from pydcop import infrastructure
from pydcop.algorithms import amaxsum
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import UnaryFunctionRelation
from pydcop.infrastructure.run import synchronous_single_run
from pydcop.infrastructure.communication import InProcessCommunicationLayer

logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum smart coloring test')

"""
Sample : Graph coloring with costs :

2 variables:
* x1 with 2 colors [0,1]
* x2 with 3 colors [0, 1, 2]

1 'all_diff' hard constraint: x1 and x2 must choose different colors

A cost is associated with each color:
* for x1:  {0: 0, 1: -3}
* for x2:   {0: 0, 1: -2, 2: -1}
Costs as modelled as unary constraints on x1 and x2.


"""



def distribue_agent_for_all(variables, factors):
    comm = infrastructure.communication.InProcessCommunicationLayer()

    node_agents = []
    for v in variables:
        f_for_variable = [f.name for f in factors if v.name in
                          [i.name for i in f.dimensions]]

        a = infrastructure.Agent('Var_' + v.name, comm)
        a.add_computation(amaxsum.VariableAlgo(v, f_for_variable))
        node_agents.append(a)

    for f in factors:
        a = infrastructure.Agent('Fact_' + f.name, comm)
        a.add_computation(amaxsum.FactorAlgo(f))
        node_agents.append(a)

    return node_agents


def graph_coloring_pb():

    # Variables and domain
    d1 = [0, 1]
    d2 = [0, 1, 2]
    x1 = Variable('x1', d1)
    x2 = Variable('x2', d2)

    # Cost functions for x1 and x2
    x1_cost = UnaryFunctionRelation('x1_cost', x1,
                                    lambda v: {0: 0, 1: -3}[v])
    x2_cost = UnaryFunctionRelation('x2_cost', x2,
                                    lambda v: {0: 0, 1: -2, 2: -1}[v])

    # Constraint x1 != x2
    # Without any cost
    @relations.AsNAryFunctionRelation(x1, x2)
    def all_diff(x1_val, x2_val):
        if x1_val == x2_val:
            return 10000
        return 0

    # Map the factor graph to agents
    variables = [x1, x2]
    factors = [x1_cost, x2_cost, all_diff]
    node_agents= distribue_agent_for_all(variables, factors)

    # and solve it
    results, _, _ = synchronous_single_run(node_agents)

    print(results)
    if results['x1'] == 1 and results['x2'] == 2:
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return graph_coloring_pb()


if __name__ == "__main__":
    sys.exit(run_test())
