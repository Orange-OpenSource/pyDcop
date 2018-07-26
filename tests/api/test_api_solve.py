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


from pydcop.dcop.objects import create_agents
from pydcop.infrastructure.run import solve
from tests.api.instances_and_utils import dcop_graphcoloring_3


def test_api_solve_maxsum():

    dcop = dcop_graphcoloring_3()
    # Agents
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'maxsum','adhoc', timeout=3)

    check_suboptimal_result(assignment)

def test_api_solve_dsa():

    dcop = dcop_graphcoloring_3()
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'dsa','oneagent', timeout=3)

    check_suboptimal_result(assignment)


def test_api_solve_dsatuto():

    dcop = dcop_graphcoloring_3()
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'dsatuto','oneagent', timeout=3)

    check_suboptimal_result(assignment)


def test_api_solve_mgm():

    dcop = dcop_graphcoloring_3()
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'mgm','oneagent', timeout=3)

    check_suboptimal_result(assignment)


def test_api_solve_mgm2():

    dcop = dcop_graphcoloring_3()
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'mgm2','oneagent', timeout=3)

    check_suboptimal_result(assignment)


def test_api_solve_dpop():

    dcop = dcop_graphcoloring_3()
    dcop.add_agents(create_agents('a', [1, 2, 3], capacity=50))

    assignment = solve(dcop, 'dpop','oneagent', timeout=3)

    check_optimal_result(assignment)


def check_optimal_result(assignment):
    assert assignment == {'v1': 'R', 'v2': 'G', 'v3': 'R'}

def check_suboptimal_result(assignment):
    # An incomplete algo does not always find the best solution but
    # finds one of two solution that does not break hard constraints.
    one_of_two = (assignment == {'v1': 'R', 'v2': 'G', 'v3': 'R'}) or \
           (assignment == {'v1': 'G', 'v2': 'R', 'v3': 'G'})
    assert one_of_two
