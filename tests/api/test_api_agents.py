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


from pydcop.dcop.objects import AgentDef, create_agents


def test_api_create_agent_minimal():

    # The name is the only mandatory param when creating an agent definition?
    a1 = AgentDef('a1')

    assert a1.name == 'a1'
    # Defaults values for route and hosting costs:
    assert a1.route('a_foo') == 1
    assert a1.hosting_cost('computation_bar') == 0


def test_api_create_agent_with_default_cost():

    a1 = AgentDef('a1', default_route=10, default_hosting_cost=5)

    assert a1.name == 'a1'
    # Defaults values for route and hosting costs:
    assert a1.route('a_foo') == 10
    assert a1.hosting_cost('computation_bar') == 5


def test_api_create_agent_with_specific_cost_as_dict():

    a1 = AgentDef('a1', routes={'a2': 8},
                  hosting_costs={'c1': 3})

    assert a1.name == 'a1'
    # Specific and defaults values for route and hosting costs:
    assert a1.route('a_foo') == 1
    assert a1.route('a2') == 8
    assert a1.hosting_cost('c1') == 3
    assert a1.hosting_cost('computation_bar') == 0


def test_api_create_several_agents():

    agents = create_agents('a', [1, 2, 3])
    assert agents['a1'].name == 'a1'
    assert 'a3' in agents


    # Agents
    # TODO : use callable for hosting and route ?

