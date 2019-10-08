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


from pydcop.dcop.dcop import DCOP, filter_dcop
from pydcop.dcop.objects import Variable, VariableDomain, AgentDef, \
    create_agents
from pydcop.dcop.relations import constraint_from_str


def test_dcop_add_constraint_is_enough():
    dcop = DCOP()

    d = VariableDomain('test', 'test' , values=range(10))
    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    c1 = constraint_from_str('c1', '0 if v1 != v2 else 1000', [v1, v2])

    dcop.add_constraint(c1)

    assert dcop.constraints == {'c1' : c1}
    assert dcop.variables == {'v1': v1, 'v2': v2}
    assert dcop.domains == {'test': d}


def test_dcop_add_constraint_easy_api():
    dcop = DCOP()
    d = VariableDomain('test', 'test' , values=range(10))

    v1 = Variable('v1', d)
    v2 = Variable('v2', d)

    dcop += 'c1', '0 if v1 != v2 else 1000', [v1, v2]

    assert 'c1' in dcop.constraints
    assert dcop.variables == {'v1': v1, 'v2': v2}
    assert dcop.domains == {'test': d}

def test_dcop_add_single_agent():
    dcop = DCOP()
    a1 = AgentDef('a1')
    dcop.add_agents(a1)
    assert dcop.agent('a1').name == 'a1'

def test_dcop_add_agents_from_list():
    dcop = DCOP()
    a1 = AgentDef('a1')
    a2 = AgentDef('a2')
    dcop.add_agents([a1, a2])
    assert dcop.agent('a1').name == 'a1'
    assert dcop.agent('a2').name == 'a2'

def test_dcop_add_agents_from_dict():
    dcop = DCOP()
    agts = create_agents('a', [1, 2, 3])
    dcop.add_agents(agts)

    assert dcop.agent('a1').name == 'a1'
    assert dcop.agent('a2').name == 'a2'
    assert dcop.agent('a3').name == 'a3'


def test_filter_dcop():
    dcop = DCOP()

    d = VariableDomain('test', 'test' , values=range(10))

    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    dcop += 'c1', '0 if v1 != v2 else 1000', [v1, v2]
    dcop.add_variable(v3)
    assert dcop.variables == {'v1': v1, 'v2': v2, "v3": v3}

    filtered = filter_dcop(dcop)

    assert filtered.variables == {'v1': v1, 'v2': v2}

def test_filter_dcop_unary_constraint():
    dcop = DCOP()

    d = VariableDomain('test', 'test' , values=range(10))

    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)

    dcop += 'c1', '0 if v1 != v2 else 1000', [v1, v2]
    # unary constraint on V3
    dcop += 'c2', 'v3 * 0.5', [v3]
    assert dcop.variables == {'v1': v1, 'v2': v2, "v3": v3}

    filtered = filter_dcop(dcop)

    # v3 is only involved in unary constraints c2 => v3 and c2 should be gone
    assert filtered.variables == {'v1': v1, 'v2': v2}
    assert "c2" not in filtered.constraints


def test_filter_dcop_unary_constraint_accepted():
    dcop = DCOP()

    d = VariableDomain('test', 'test' , values=range(10))

    v1 = Variable('v1', d)
    v2 = Variable('v2', d)
    v3 = Variable('v3', d)
    v4 = Variable('v4', d)

    dcop += 'c1', '0 if v1 != v2 else 1000', [v1, v2]
    # unary constraint on V3
    dcop += 'c2', 'v3 * 0.5', [v3]
    dcop.add_variable(v4)

    filtered = filter_dcop(dcop, accept_unary=True)

    # c2 and v3 must stay here, as we accept unary constraints, but v4 must be gone
    assert "v3" in filtered.variables
    assert "c2" in filtered.constraints
    assert "v4"  not in filtered.variables
