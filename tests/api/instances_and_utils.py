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


"""Various utils and tests instances for API testing.

"""
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, create_variables


def dcop_graphcoloring_3():
    """
    Build a simple 3-variable graph coloring problem.

    v1--v2--v3

    Each variable has a cost function which makes it prefer one color:
    v1 prefers R
    v2 prefers G
    v3 prefers G

    Of course, the preferences of v2 and v3 conflict.
    The best affectation is
    v1 - R
    v2 - G
    v3 - R

    Returns
    -------

    """
    dcop = DCOP('graphcoloring_3')
    # Domain and variables
    d = Domain('color', '', ['R', 'G'])
    variables = create_variables('v', ['1', '2', '3'], d)
    # unary constraints for preferences
    dcop += 'cost_1', '-0.1 if v1 == "R" else 0.1 ', variables
    dcop += 'cost_2', '-0.1 if v2 == "G" else 0.1 ', variables
    dcop += 'cost_3', '-0.1 if v3 == "G" else 0.1 ', variables
    # coloring constraints : v1 != v2 != v3
    dcop += 'c1', '1 if v1 == v2 else 0', variables
    dcop += 'c2', '1 if v3 == v2 else 0', variables

    # Note that we do not define the agents for this dcop here,
    # as the number of agents depends on the test case we will use the dcop for.

    return dcop
