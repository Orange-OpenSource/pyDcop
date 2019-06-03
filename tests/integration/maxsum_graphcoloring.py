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

from pydcop.algorithms import amaxsum
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.run  import synchronous_single_run
from pydcop.infrastructure.communication import InProcessCommunicationLayer

# Graph Coloring example:
# This example is the same as the one used in MaxSum paper, it's a simple
# graph coloring problem with three agent and two colors :
#    a1---a2---a3
# We use a variant of this problems in which each variable has some
# preferences about its favorite color.

# Without preferences, this problem has a strong symmetry and several optimal
# solutions: applying MaxSum does not allow to make a choice and each variable
# keeps its full domain available, which means that for any value di from
# domain Di of variable xi, there exists an optimal assignment in which
# xi take di. This is demonstrated in graph_coloring_no_prefs().
#
# Adding the preferences breaks the symmetry and MaxSum correctly find the
# optimal assignment.

# We model the graph coloring problem as 3 agents with each one variable
# and one factor, meaning we have 3 variables and three factors.

# Factor Graph :
#
#    u1 ------ x2 ----- u3
#     |        |        |
#    x1 ------ u2 ----- x3
#
#  Agent a(n) hosts u(n) and v(n)

logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum Smart Lighting test 1')


def distribute_agents(var_facts):
    comm = InProcessCommunicationLayer()

    dcop_agents = []
    factors = [f for _, f in var_facts]

    i = 0
    for v, f in var_facts:

        # Algorithm for variable
        # build the list of factors that depend on this variable
        f_for_variable = [f.name for f in factors if v.name in
                          [i.name for i in f.dimensions]]
        v_a = amaxsum.VariableAlgo(v, f_for_variable)

        # Algorithm for the factor
        f_a = amaxsum.FactorAlgo(f)

        # Agent hosting the factor and variable
        a = Agent('a_'+str(i), comm)
        a.add_computation(v_a)
        a.add_computation(f_a)
        i += 1

        dcop_agents.append(a)

    return dcop_agents, comm


def graph_coloring_with_prefs():
    """
    Create variables and factors for a sample graph coloring problem.

    :return: a list of (variable, factor) pairs
    """
    # Variables and domain
    d = ['R', 'G']
    x1 = Variable('x1', d)
    x2 = Variable('x2', d)
    x3 = Variable('x3', d)

    # Preferences for each variable :
    # x1 prefers 'R' p(v1) = [ -0.1, +0.1]
    # x2 and x3 prefer 'G'  p(v2) = p(x3) = [ +0.1, -0.1]
    def p1(v):
        return -0.1 if v == 'R' else 0.1

    def p2(v):
        return -0.1 if v == 'G' else 0.1

    def p3(v):
        return -0.1 if v == 'G' else 0.1

    # Factors u1, u2 and u3 for the three agents
    @relations.AsNAryFunctionRelation(x1, x2)
    def u1(x1_, x2_):
        c = 1 if x1_ == x2_ else 0
        return c + p1(x1_) + p2(x2_)

    @relations.AsNAryFunctionRelation(x1, x2, x3)
    def u2(x1_, x2_, x3_):
        c = 1 if x1_ == x2_ else 0
        c += 1 if x2_ == x3_ else 0
        return c + p1(x1_) + p2(x2_) + p3(x3_)

    @relations.AsNAryFunctionRelation(x2, x3)
    def u3(x2_, x3_):
        c = 1 if x2_ == x3_ else 0
        return c + p2(x2_) + p3(x3_)

    # Map the factor graph to agents
    node_agents, comm = distribute_agents([(x1, u1), (x2, u2), (x3, u3)])

    # and solve it
    results, _, _ = synchronous_single_run(node_agents)

    if results['x1'] == 'R' and results['x2'] == 'G' and results['x3'] == 'R':
        logging.info('SUCCESS !! ')
        return 0
    else:
        logging.info('invalid result found, needs some debugging ...' + str(
            results))
        return 1


def run_test():
    return graph_coloring_with_prefs()


if __name__ == "__main__":
    sys.exit(graph_coloring_with_prefs())
