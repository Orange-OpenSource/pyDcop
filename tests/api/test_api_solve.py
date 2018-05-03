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


from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, create_variables, create_agents
from pydcop.infrastructure.run import solve
from tests.api.instances_and_utils import dcop_graphcoloring_3


def test_api_solve_maxsum():

    dcop = dcop_graphcoloring_3()
    # Agents
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'maxsum','adhoc', timeout=3)

    assert assignment['v1'] == 'R'
    assert assignment['v2'] == 'G'
    assert assignment['v3'] == 'R'


def test_api_solve_dsa():


    dcop = dcop_graphcoloring_3()
    # Agents
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'dsa','adhoc', timeout=3)

    # Do not check values, DSA is quite bad on this pb


def test_api_solve_dpop():


    dcop = dcop_graphcoloring_3()
    # Agents
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'dpop','oneagent', timeout=3)

    assert assignment['v1'] == 'R'
    assert assignment['v2'] == 'G'
    assert assignment['v3'] == 'R'
