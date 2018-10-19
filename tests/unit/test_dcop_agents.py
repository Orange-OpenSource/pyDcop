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
from pydcop.utils.simple_repr import from_repr, simple_repr


def test_name_only():

    a = AgentDef("a1")
    assert a.name == "a1"


def test_with_capacity():
    a = AgentDef("a1", capacity=10)

    assert a.name == "a1"
    assert a.capacity == 10


def test_with_arbitrary_attr():

    a = AgentDef("a1", foo=15)

    assert a.name == "a1"
    assert a.foo == 15


def test_with_various_attr():

    a = AgentDef("a1", foo=15, bar="bar")

    assert a.name == "a1"
    assert a.foo == 15
    assert a.bar == "bar"


def test_with_default_hosting_cost():

    a = AgentDef("a1", default_hosting_cost=15)

    assert a.name == "a1"
    assert a.hosting_cost("foo") == 15


def test_with_hosting_cost():

    a = AgentDef("a1", hosting_costs={"foo": 5, "bar": 7})

    assert a.name == "a1"
    assert a.hosting_cost("foo") == 5
    assert a.hosting_cost("bar") == 7
    assert a.hosting_cost("pouet") == 0


def test_with_default_route():

    a = AgentDef("a1", default_route=12)

    assert a.name == "a1"
    assert a.route("foo") == 12
    assert a.route("bar") == 12


def test_with_routes():

    a = AgentDef("a1", routes={"psycho": 5, "killer": 7})

    assert a.name == "a1"
    assert a.route("psycho") == 5
    assert a.route("killer") == 7
    assert a.route("ahahah") == 1


def test_create_agents_from_range():
    agts = create_agents("a", range(20), default_route=2, default_hosting_costs=7)
    assert isinstance(agts["a08"], AgentDef)
    assert len(agts) == 20


def test_create_agents_with_capacity():
    agts = create_agents("a", range(20), capacity=100)
    assert isinstance(agts["a18"], AgentDef)
    assert agts["a05"].capacity == 100
    assert len(agts) == 20
