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

import pydcop.infrastructure.communication
from pydcop import infrastructure
from pydcop.algorithms import amaxsum
from pydcop.dcop import relations
from pydcop.dcop.objects import Variable

logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum Smart Lighting test 1')

# Graph Coloring example:
# This example is the same as the one used in MaxSum paper, it's a simple
# graph coloring problem with three agent and two colors :
#    a1---a2---a3
# There is one variant for this problems in which each variable has some
# preferences about its favorite color.
#
# Without preferences, this problem has a strong symmetry and several optimal
# solutions: applying maxsum does not allow to make a choice and each variable
# keeps its full domain available, which means that for any value di from
# domain Di of variable xi, there exists an optimal assignment in which
# xi take di. This is demonstrated in graph_coloring_no_prefs().
#
# Adding the preferences breaks the symmetry and MaxSum correctly find the
# optimal assignment.

# In these samples, factor and variable have each their own agent.
# This is different than the distribution presented in the MaxSum paper where
# a pair (variable, factor) was assigned to each agent.

# Model:
# There are several ways of transforming this graph coloring problem into a
# factor graph.
# * Using one factor for each constraint and one variable for each agent in the
#   graph, resulting in 3 variables and 2 factors
#   * The advantage of this approach is that it produces less cycle in the graph
#   * The problem is that there is no obvious distribution of the computation to
#     physical nodes of the system.
#   * easier to have binary factors
#   * if the graph is a tree (no cycle), optimal solution is found and the
#     algorithm terminates
# * Using one (variable, factor) pair for each agent.
#   * obvious mapping to physical nodes
#   * but the graph will (almost) always have loops, meaning that MaxSum will
#     only produce an approximation of the optimal solution.
#   * more difficult to have binary factors


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


def graph_coloring_no_prefs():

    # Extremely simple graph coloring problem
    # Three variables, 2 constraints, no cycle
    # modelled as 3 variables and two factors, each variable and each factor
    # has its own agent.

    # This problem has a strong symmetry and several optimal solutions:
    # applying maxsum does not allow to make a choice and each variable keeps
    # its full domain available, which means that for any value di from
    # domain Di of variable xi, there exists an optimal assignment in which
    # xi take di.
    #
    # To select a solution, an extra step would be needed : select a value
    # for one variable and walk the tree to get the value for other variables
    # induced by this first choice.

    d = ['R', 'G']
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    # cost table for the factor, simply reflects the fact that neighbor
    # variables shoud not have the same value
    BIN_DIFF_TABLE_2 = [[1, 0],
                        [0, 1]]
    # Factor between v1 and v2
    f12 = relations.NAryMatrixRelation([v1, v2], name='f12',
                                       matrix=BIN_DIFF_TABLE_2)

    # Factor between v2 and v3
    f23 = relations.NAryMatrixRelation([v2, v3], name='f23',
                                       matrix=BIN_DIFF_TABLE_2)

    # Map the factor graph to agents and solve it
    node_agents = distribue_agent_for_all([v1, v2, v3], [f12, f23])
    # this sample does not work with our standard run method :
    # no factor is a leaf, which means that no start message is sent !
    # We need to initiate messages manually
    for n in node_agents:
        for a in n.computations:
            if hasattr(a, '_init_msg'):
                a._init_msg()
    res, _, _ = infrastructure.synchronous_single_run(node_agents)

    print("FINISHED ! " + str(res))


def graph_coloring_with_prefs():

    # In this setup, we introduce preferences for each variable
    # as we are minimizing, we express preferences as cost (with lower cost
    # for preferred value)
    # v1 prefers 'R' p(v1) = [ -0.1, +0.1]
    # v2 and v3 prefer 'G'  p(v2) = p(v3) = [ +0.1, -0.1]

    # With This preferences, MaxSum now correctly selects the optimal assignment
    # v1 = R, v2 = G , v3 = R

    d = ['R', 'G']
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    # Factor (cost) between v1 and v2
    # f12 = (x1 == x2) + p(x1) + p(x2)
    # where (x1 == x2) is 1 if x1 equals x2 and 0 otherwise
    BIN_DIFF_TABLE_PREFS_12 = [[1, 0.2],
                               [-0.1, 1]]

    f12 = relations.NAryMatrixRelation([v1, v2], name='f12',
                                       matrix=BIN_DIFF_TABLE_PREFS_12)
    # f12 = maxsum.BinaryValueTable('f12', v1, v2, BIN_DIFF_TABLE_PREFS_12)

    # Factor between v2 and v3
    BIN_DIFF_TABLE_PREFS_23 = [[1.2, 0],
                               [0, 0.8]]
    f23 = relations.NAryMatrixRelation([v2, v3], name='f23',
                                       matrix=BIN_DIFF_TABLE_PREFS_23)
    # f23 = maxsum.BinaryValueTable('f23', v2, v3, BIN_DIFF_TABLE_PREFS_23)

    # Map the factor graph to agents and solve it
    node_agents = distribue_agent_for_all([v1, v2, v3], [f12, f23])
    # this sample does not work with our standard run method :
    # no factor is a leaf, which means that no start message is sent !
    # We need to initiate messages manually
    for n in node_agents:
        for a in n.computations:
            if hasattr(a, '_init_msg'):
                a._init_msg()
    res, _, _ = infrastructure.synchronous_single_run(node_agents)

    print("FINISHED ! " + str(res))


def run_test():
    graph_coloring_with_prefs()
    graph_coloring_no_prefs()

    # We do not really check the result found by the algorithm here
    # but still running this as an integration test at least makes sure that we
    # did not break something basic.
    return 0

if __name__ == "__main__":
    #sys.exit(dpop_petcu())


    # TODO select sample to run on the cli
    # FIXME : this sample does not work with our standard run method :
    # no factor is a leaf, which means that no start message is sent !
    graph_coloring_with_prefs()
    # graph_coloring_no_prefs()