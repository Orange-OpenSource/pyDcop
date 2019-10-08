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


from pydcop.algorithms import (
    AlgorithmDef,
    list_available_algorithms,
    load_algorithm_module,
)
from pydcop.utils.simple_repr import simple_repr, from_repr


def test_algo_def():

    a = AlgorithmDef("maxsum", {"stability": 0.01}, "min")

    assert a.algo == "maxsum"
    assert a.mode == "min"
    assert "stability" in a.param_names()
    assert a.param_value("stability") == 0.01


def test_simple_repr():

    a = AlgorithmDef("maxsum", {"stability": 0.01}, "min")

    r = simple_repr(a)

    assert r["algo"] == "maxsum"
    assert r["mode"] == "min"
    assert r["params"]["stability"] == 0.01


def test_from_repr():

    a = AlgorithmDef("maxsum", {"stability": 0.01}, "min")

    r = simple_repr(a)
    a2 = from_repr(r)

    assert a == a2
    assert a2.param_value("stability") == 0.01


def test_building_algodef_with_default_params():

    a = AlgorithmDef.build_with_default_param("amaxsum")

    assert a.params["damping"] == 0.5


def test_building_algodef_with_provided_and_default_params():

    a = AlgorithmDef.build_with_default_param("dsa", {"variant": "B"}, mode="max")

    assert a.params["variant"] == "B"  # provided param
    assert a.params["probability"] == 0.7  # default param
    assert a.algo == "dsa"
    assert a.mode == "max"


def test_load_algorithm():

    # We test load for all available algorithms
    for a in list_available_algorithms():
        algo = load_algorithm_module(a)

        assert algo.algorithm_name == a
        assert hasattr(algo, "communication_load")
        assert hasattr(algo, "computation_memory")


def test_load_algorithm_with_default_footprint():

    # dsatuto has no load method defined : check that we get instead default
    # implementations
    algo = load_algorithm_module("dsatuto")
    assert algo.algorithm_name == "dsatuto"
    assert algo.communication_load(None, None) == 1
    assert algo.computation_memory(None) == 1
