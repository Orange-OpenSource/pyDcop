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

"""
Unit tests for the SyncBB algorithm.

"""
import pytest

from pydcop.algorithms.syncbb import get_value_candidates, get_next_assignment
from pydcop.computations_graph.pseudotree import build_computation_graph
from pydcop.dcop.objects import Domain, Variable
from pydcop.dcop.relations import constraint_from_str



def test_get_candidates_no_value_selected():
    d = Domain("d", "vals", [0, 1, 2, 3])
    v = Variable("v", d)

    obtained = get_value_candidates(v, None)
    assert obtained == [0, 1, 2, 3]


def test_get_candidate_value_selected():
    d = Domain("d", "vals", ["B", "D", "A", "E"])
    v = Variable("v", d)

    obtained = get_value_candidates(v, "B")
    assert obtained == ["D", "A", "E"]

    obtained = get_value_candidates(v, "A")
    assert obtained == ["E"]

    obtained = get_value_candidates(v, "E")
    assert obtained == []

