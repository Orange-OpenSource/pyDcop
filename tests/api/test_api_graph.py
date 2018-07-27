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


from time import sleep

import pydcop.computations_graph.constraints_hypergraph as chg

from pydcop.algorithms import AlgorithmDef, ComputationDef
from pydcop.dcop.objects import Domain, Variable
from pydcop.dcop.relations import constraint_from_str
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer
from pydcop.infrastructure.computations import build_computation


def test_api_cg_creation_dsa():
    # creating a computation graph from API, without relying on a full
    # description of the DCOP.
    # Two API level
    #  * DCOP : create a dcop
    #  * Computation graph:

    # DCOP
    # works when using an orchestrator that can transform a dcop into a
    # computation graph and distribute the computation on the set of agents.
    # Efficient and simple solution when there is a single deployement of the
    # system at startup.

    # Computation Graph
    # create the computations directly
    # Necessary when there is no central place where the full dcop could be
    # created. Each agent build the computation it will run, for example when
    # the definition of the dcop changes at run time or when a new dcop must be
    # created and solve completely dynamically and runtime without relying on a
    # central element like an orchestrator.

    # to create a computation instance, one need:
    # of course, variable(s) and constraint(s) like for the dcop.
    # but also
    # an algo_def : given as input
    # a comp_node : depends of the type of computation graph, requires a
    # variable and  or constraints

    # Here we define a simple graph coloring problem with 3 variables:
    d = Domain('color', '', ['R', 'G'])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    # We need to define all agents first, because we will need their address
    # when registering neighbors computation
    agt1 = Agent('a1', InProcessCommunicationLayer())
    agt2 = Agent('a2', InProcessCommunicationLayer())
    agt3 = Agent('a3', InProcessCommunicationLayer())

    # Agent 1 with Variable v1
    # Constraints in which v1 is involved
    cost_v1 = constraint_from_str('cv1', '-0.1 if v1 == "R" else 0.1 ', [v1])
    diff_v1_v2 = constraint_from_str('c1', '1 if v1 == v2 else 0', [v1, v2])
    # Computation node for the variable with these constraints
    node_v1 = chg.VariableComputationNode(v1, [cost_v1, diff_v1_v2])
    comp_def = ComputationDef(node_v1, AlgorithmDef.build_with_default_param('dsa'))
    v1_computation = build_computation(comp_def)
    # and register the computation and the agents for the neighboring
    # computations.
    agt1.add_computation(v1_computation)
    agt1.discovery.register_computation('v2', 'a2', agt2.address)

    # Agent 2 with Variable v2
    cost_v2 = constraint_from_str('cv2', '-0.1 if v2 == "G" else 0.1 ', [v2])
    diff_v2_v3 = constraint_from_str('c2', '1 if v2 == v3 else 0', [v2, v3])

    node_v2 = chg.VariableComputationNode(v2, [cost_v2, diff_v2_v3, diff_v1_v2])
    comp_def_v2 = ComputationDef(node_v2,
                                 AlgorithmDef.build_with_default_param('dsa'))
    v2_computation = build_computation(comp_def_v2)

    agt2.add_computation(v2_computation)
    agt2.discovery.register_computation('v1', 'a1', agt1.address)
    agt2.discovery.register_computation('v3', 'a3', agt3.address)

    # Agent 3 with Variable v3
    cost_v3 = constraint_from_str('cv3', '-0.1 if v3 == "G" else 0.1 ', [v3])

    node_v3 = chg.VariableComputationNode(v3, [cost_v3, diff_v2_v3])
    comp_def_v3 = ComputationDef(node_v3,
                                 AlgorithmDef.build_with_default_param('dsa'))
    v3_computation = build_computation(comp_def_v3)

    agt3.add_computation(v3_computation)
    agt3.discovery.register_computation('v2', 'a2', agt2.address)

    # Start and run the 3 agents manually:
    agts = [agt1, agt2, agt3]
    for a in agts:
        a.start()
    for a in agts:
        a.run()
    sleep(1)  # let the system run for 1 second
    for a in agts:
        a.stop()

    # As we do not have an ochestrator, we need to collect results manually:
    assert agt1.computation('v1').current_value != \
        agt2.computation('v2').current_value
    assert agt3.computation('v3').current_value != \
        agt2.computation('v2').current_value


def test_api_cg_creation_mgm():
    # This time we solve the same graph coloring problem with the MGM algorithm.
    # As you can see, the only difference is the creation of computation with
    # mgm.build_computation
    d = Domain('color', '', ['R', 'G'])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    # We need to define all agents first, because we will need their address
    # when registering neighbors computation
    agt1 = Agent('a1', InProcessCommunicationLayer())
    agt2 = Agent('a2', InProcessCommunicationLayer())
    agt3 = Agent('a3', InProcessCommunicationLayer())

    # Agent 1 with Variable v1
    cost_v1 = constraint_from_str('cv1', '-0.1 if v1 == "R" else 0.1 ', [v1])
    diff_v1_v2 = constraint_from_str('c1', '1 if v1 == v2 else 0', [v1, v2])

    node_v1 = chg.VariableComputationNode(v1, [cost_v1, diff_v1_v2])
    comp_def = ComputationDef(node_v1,
                              AlgorithmDef.build_with_default_param('mgm'))
    v1_computation = build_computation(comp_def)

    agt1.add_computation(v1_computation)
    # We need to register manually as we are not using the directory hosted by
    # the orchestrator.
    agt1.discovery.register_computation('v2', 'a2', agt2.address)

    # Agent 2 with Variable v2
    cost_v2 = constraint_from_str('cv2', '-0.1 if v2 == "G" else 0.1 ', [v2])
    diff_v2_v3 = constraint_from_str('c2', '1 if v2 == v3 else 0', [v2, v3])

    node_v2 = chg.VariableComputationNode(v2, [cost_v2, diff_v2_v3, diff_v1_v2])
    comp_def_v2 = ComputationDef(node_v2,
                                 AlgorithmDef.build_with_default_param('mgm'))
    v2_computation = build_computation(comp_def_v2)

    agt2.add_computation(v2_computation)
    agt2.discovery.register_computation('v1', 'a1', agt1.address)
    agt2.discovery.register_computation('v3', 'a3', agt3.address)

    # Agent 3 with Variable v3
    cost_v3 = constraint_from_str('cv3', '-0.1 if v3 == "G" else 0.1 ', [v3])

    node_v3 = chg.VariableComputationNode(v3, [cost_v3, diff_v2_v3])
    comp_def_v3 = ComputationDef(node_v3,
                                 AlgorithmDef.build_with_default_param('mgm'))
    v3_computation = build_computation(comp_def_v3)

    agt3.add_computation(v3_computation)
    agt3.discovery.register_computation('v2', 'a2', agt2.address)

    # Start and run the 3 agents manually:
    agts = [agt1, agt2, agt3]
    for a in agts:
        a.start()
    for a in agts:
        a.run()
    sleep(1)  # let the system run for 1 second
    for a in agts:
        a.stop()

    # As we do not have an orchestrator, we need to collect results manually:
    # With MGM we will not always be able to get the optimal solution,
    # depending on the start affectation (which may require a coordinated
    # change and thus is not possible with mgm), so we just check the
    # hard constraints
    assert agt1.computation('v1').current_value != \
        agt2.computation('v2').current_value
    assert agt3.computation('v3').current_value != \
        agt2.computation('v2').current_value


def test_api_cg_creation_mgm2():
    # This time we solve the same graph coloring problem with the MGM2
    # algorithm.
    # As you can see, the only difference is the creation of the AlgorithmDef
    # instance with 'mgm2'
    d = Domain('color', '', ['R', 'G'])
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    # We need to define all agents first, because we will need their address
    # when registering neighbors computation
    agt1 = Agent('a1', InProcessCommunicationLayer())
    agt2 = Agent('a2', InProcessCommunicationLayer())
    agt3 = Agent('a3', InProcessCommunicationLayer())

    # Agent 1 with Variable v1
    cost_v1 = constraint_from_str('cv1', '-0.1 if v1 == "R" else 0.1 ', [v1])
    diff_v1_v2 = constraint_from_str('c1', '1 if v1 == v2 else 0', [v1, v2])

    node_v1 = chg.VariableComputationNode(v1, [cost_v1, diff_v1_v2])
    comp_def = ComputationDef(node_v1,
                              AlgorithmDef.build_with_default_param('mgm2'))
    v1_computation = build_computation(comp_def)

    agt1.add_computation(v1_computation)
    # We need to register manually as we are not using the directory hosted by
    # the orchestrator.
    agt1.discovery.register_computation('v2', 'a2', agt2.address)

    # Agent 2 with Variable v2
    cost_v2 = constraint_from_str('cv2', '-0.1 if v2 == "G" else 0.1 ', [v2])
    diff_v2_v3 = constraint_from_str('c2', '1 if v2 == v3 else 0', [v2, v3])

    node_v2 = chg.VariableComputationNode(v2, [cost_v2, diff_v2_v3, diff_v1_v2])
    comp_def_v2 = ComputationDef(node_v2,
                                 AlgorithmDef.build_with_default_param('mgm2'))
    v2_computation = build_computation(comp_def_v2)

    agt2.add_computation(v2_computation)
    agt2.discovery.register_computation('v1', 'a1', agt1.address)
    agt2.discovery.register_computation('v3', 'a3', agt3.address)

    # Agent 3 with Variable v3
    cost_v3 = constraint_from_str('cv3', '-0.1 if v3 == "G" else 0.1 ', [v3])

    node_v3 = chg.VariableComputationNode(v3, [cost_v3, diff_v2_v3])
    comp_def_v3 = ComputationDef(node_v3,
                                 AlgorithmDef.build_with_default_param('mgm2'))
    v3_computation = build_computation(comp_def_v3)

    agt3.add_computation(v3_computation)
    agt3.discovery.register_computation('v2', 'a2', agt2.address)

    # Start and run the 3 agents manually:
    agts = [agt1, agt2, agt3]
    for a in agts:
        a.start()
    for a in agts:
        a.run()
    sleep(1)  # let the system run for 1 second
    for a in agts:
        a.stop()

    # As we do not have an orchestrator, we need to collect results manually.
    # As with MGM, MGM2 does not necessarily find the optimal solution,
    # depending on the start affectation (which may require a 3-coordinated
    # change and thus is not possible with mgm2), so we just check the
    # hard constraints
    assert agt1.computation('v1').current_value != \
        agt2.computation('v2').current_value
    assert agt3.computation('v3').current_value != \
        agt2.computation('v2').current_value
