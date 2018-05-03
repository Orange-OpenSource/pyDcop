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


from pydcop.dcop.objects import create_binary_variables, BinaryVariable
from pydcop.reparation import create_computation_hosted_constraint, \
    create_agent_capacity_constraint, create_agent_hosting_constraint, \
    create_agent_comp_comm_constraint


def test_create_hosted_constraint_for_computation():

    # say we have a computation c1 that can be hosted on agents a1, a2 and a3
    bin_vars = create_binary_variables('v_', (['c1'], ['a1', 'a2', 'a3']))

    constraint = create_computation_hosted_constraint('c1', bin_vars)

    # check the constraints depends on one variable for each agent
    assert set(v.name for v in constraint.dimensions) == \
           {'v_c1_a1', 'v_c1_a2', 'v_c1_a3'}

    # Check some return value for the constraint
    assert constraint(v_c1_a1=1, v_c1_a2=0, v_c1_a3=0) == 0
    assert constraint(v_c1_a1=1, v_c1_a2=0, v_c1_a3=1) != 0
    assert constraint(v_c1_a1=1, v_c1_a2=1, v_c1_a3=1) != 0


def test_create_capacity_constraint_for_agent():

    # Say our agent a1 has a remaining capacity of 50 and could hosts
    # candidates computations from [c1, c2, c3, c4], which all have a
    # footprint of 25
    candidates = ['c1', 'c2', 'c3', 'c4']

    def footprint(comp_name):
        return 25

    bin_vars = create_binary_variables('x_', (candidates, ['a1']))

    capa_c = create_agent_capacity_constraint('a1', 50, footprint,
                                              bin_vars)

    assert set(v.name for v in capa_c.dimensions) == \
           {'x_c1_a1', 'x_c2_a1', 'x_c3_a1', 'x_c4_a1'}

    assert capa_c(x_c1_a1=1, x_c2_a1=0, x_c3_a1=0, x_c4_a1=0) == 0
    assert capa_c(x_c1_a1=1, x_c2_a1=1, x_c3_a1=0, x_c4_a1=0) == 0
    assert capa_c(x_c1_a1=1, x_c2_a1=1, x_c3_a1=1, x_c4_a1=0) != 0
    assert capa_c(x_c1_a1=1, x_c2_a1=1, x_c3_a1=1, x_c4_a1=1) != 0


def test_create_hosting_constraint_for_agent():

    # Say agent a1 could host the candidate computations [c1, c2, c3]
    candidates = ['c1', 'c2', 'c3']

    def hosting_cost_a1(comp_name):
        costs = {'c1': 10, 'c2': 0, 'c3': 100}
        return costs[comp_name]

    bin_vars = create_binary_variables('x_', (candidates, ['a1']))

    cost_c = create_agent_hosting_constraint('a1', hosting_cost_a1,
                                              bin_vars)

    assert set(v.name for v in cost_c.dimensions) == \
           {'x_c1_a1', 'x_c2_a1', 'x_c3_a1', }

    assert cost_c(x_c1_a1=1, x_c2_a1=0, x_c3_a1=0) == 10
    assert cost_c(x_c1_a1=1, x_c2_a1=1, x_c3_a1=0) == 10
    assert cost_c(x_c1_a1=1, x_c2_a1=1, x_c3_a1=1) == 110
    assert cost_c(x_c1_a1=0, x_c2_a1=1, x_c3_a1=1) == 100


def test_create_comm_constraint_for_agent_single_var():

    # Say 1 candidate computation c1 could be hosted on a1
    # c1 depends on c3 and c4, c3 is not a candidate (ie is fixed). c4 is a
    # candidate computation but cannot be h

    repair_info = (['a1', 'a2'],
                   {'c3': 'a2'},
                   {'c4': ['a2', 'a3']})

    # cost is uniform, no matter the computations and the agents
    def comm(com1, comp2, agt2):
        return 10

    # binary vars for the candidate var that could be hosted on a1
    bin_vars =  create_binary_variables('x_', (['c1'],
                                               ['a1']))

    # and bin vars for all the candidiate neighbors of these candidate
    _, _, mobile = repair_info
    for vm, ams in mobile.items():
        bin_vm = create_binary_variables('x_', ([vm], ams))
        bin_vars.update(bin_vm)

    comm_c = create_agent_comp_comm_constraint('a1', 'c1', repair_info,
                                          comm, bin_vars)


    assert set(v.name for v in comm_c.dimensions) == \
           {'x_c1_a1', 'x_c4_a2', 'x_c4_a3'}

    # if a1 hosts c1, the comm host with c3 and c4 will always be 20
    assert comm_c(x_c1_a1=1, x_c4_a2=1, x_c4_a3=0) == 20
    assert comm_c(x_c1_a1=1, x_c4_a2=0, x_c4_a3=1) == 20
    # and if ot does not host it, it must obvouisly be zero
    assert comm_c(x_c1_a1=0, x_c4_a2=0, x_c4_a3=1) == 0
    assert comm_c(x_c1_a1=0, x_c4_a2=1, x_c4_a3=0) == 0

