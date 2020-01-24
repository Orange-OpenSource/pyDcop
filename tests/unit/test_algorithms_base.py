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


import unittest

import numpy as np
import pytest

import pydcop.algorithms
import pydcop.dcop.relations
import pydcop.utils
import pydcop.utils.various
from pydcop.infrastructure.computations import Message
from pydcop.dcop.objects import VariableDomain, Variable, VariableWithCostFunc
from pydcop.dcop.relations import (
    UnaryFunctionRelation,
    constraint_from_str,
    assignment_cost,
    find_optimal,
    AsNAryFunctionRelation,
    optimal_cost_value,
)
from pydcop.utils.simple_repr import simple_repr, from_repr


class MessageTests(unittest.TestCase):
    def test_simple_repr(self):
        msg = Message("test", "str_content")

        r = simple_repr(msg)

        self.assertEqual(r["msg_type"], "test")
        self.assertEqual(r["content"], "str_content")

    def test_from_repr(self):
        msg = Message("test", "str_content")

        r = simple_repr(msg)
        msg2 = from_repr(r)

        self.assertEqual(msg, msg2)


class VariableDomainTest(unittest.TestCase):
    def test_domain_iterator(self):
        r = list(range(3))
        d = VariableDomain("luminosity", "luminosity", r)

        for value in d:
            self.assertTrue(value in r)

    def test_domain_in(self):
        r = list(range(3))
        d = VariableDomain("luminosity", "luminosity", r)

        for value in r:
            self.assertTrue(value in d)

    def test_domain_len(self):
        r = list(range(3))
        d = VariableDomain("luminosity", "luminosity", r)

        self.assertEqual(3, len(d))

    def test_domain_content(self):
        r = list(range(3))
        d = VariableDomain("luminosity", "luminosity", r)

        self.assertEqual(d[0], r[0])
        self.assertEqual(d[1], r[1])
        self.assertEqual(d[2], r[2])


class GenerateAssignementTestCase(unittest.TestCase):
    def test_generate_1var(self):

        x1 = Variable("x1", ["a", "b", "c"])

        ass = list(pydcop.dcop.relations.generate_assignment([x1]))

        self.assertEqual(len(ass), len(x1.domain))
        self.assertIn(["a"], ass)
        self.assertIn(["b"], ass)
        self.assertIn(["c"], ass)

    def test_generate_1var_generator(self):
        x1 = Variable("x1", ["a", "b", "c"])

        ass = pydcop.dcop.relations.generate_assignment([x1])

        res = [["a"], ["b"], ["c"]]
        for a in ass:
            print(a)
            self.assertIn(a, res)
            res.remove(a)
        # make sure we have yield all possible combinations
        self.assertEqual(len(res), 0)

    def test_generate_2var(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["a", "b", "c"])

        ass = list(pydcop.dcop.relations.generate_assignment([x1, x2]))
        print(ass)
        self.assertEqual(len(ass), len(x1.domain) * len(x2.domain))
        self.assertIn(["a", "a"], ass)
        self.assertIn(["b", "c"], ass)
        self.assertIn(["c", "a"], ass)

    def test_generate_3var(self):
        x1 = Variable("x1", ["a1", "a2", "a3"])
        x2 = Variable("x2", ["b1"])
        x3 = Variable("x3", ["c1", "c2"])

        ass = list(pydcop.dcop.relations.generate_assignment([x1, x2, x3]))

        self.assertEqual(len(ass), len(x1.domain) * len(x2.domain) * len(x3.domain))
        self.assertIn(["a1", "b1", "c2"], ass)
        self.assertIn(["a3", "b1", "c1"], ass)
        self.assertIn(["a2", "b1", "c2"], ass)


class GenerateAssignementAsDictTestCase(unittest.TestCase):
    def test_generate_1var(self):
        x1 = Variable("x1", ["a", "b", "c"])

        ass = list(pydcop.dcop.relations.generate_assignment_as_dict([x1]))

        self.assertEqual(len(ass), len(x1.domain))
        self.assertIn({"x1": "a"}, ass)
        self.assertIn({"x1": "b"}, ass)
        self.assertIn({"x1": "c"}, ass)

    def test_generate_2var(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["a", "b", "c"])

        ass = list(pydcop.dcop.relations.generate_assignment_as_dict([x1, x2]))
        print(ass)
        self.assertEqual(len(ass), len(x1.domain) * len(x2.domain))
        self.assertIn({"x1": "a", "x2": "a"}, ass)
        self.assertIn({"x1": "b", "x2": "c"}, ass)
        self.assertIn({"x1": "c", "x2": "a"}, ass)


class FindArgOptimalTestCase(unittest.TestCase):
    def test_findargmax(self):
        # u1 is a relation with a single variable :
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = pydcop.dcop.relations.NAryMatrixRelation(
            [x1], np.array([2, 4, 8], np.int8)
        )

        # take the projection of u1 along x1
        m, c = pydcop.dcop.relations.find_arg_optimal(x1, u1, mode="max")

        self.assertEqual(len(m), 1)
        self.assertEqual(m[0], "c")
        self.assertEqual(c, 8)

    def test_findargmin(self):
        # u1 is a relation with a single variable :
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = pydcop.dcop.relations.NAryMatrixRelation(
            [x1], np.array([2, 4, 8], np.int8)
        )

        # take the projection of u1 along x1
        m, c = pydcop.dcop.relations.find_arg_optimal(x1, u1, mode="min")

        self.assertEqual(len(m), 1)
        self.assertEqual(m[0], "a")
        self.assertEqual(c, 2)

    def test_findargmin_fct(self):

        v1 = Variable("v1", list(range(10)))
        f1 = UnaryFunctionRelation("f1", v1, lambda x: abs(x - 5))

        m, c = pydcop.dcop.relations.find_arg_optimal(v1, f1, mode="min")

        self.assertEqual(len(m), 1)
        self.assertEqual(m[0], 5)
        self.assertEqual(c, 0)

    def test_findargmin_several_values(self):
        v1 = Variable("v1", list(range(10)))
        f1 = UnaryFunctionRelation("f1", v1, lambda x: 2 if 3 < x < 6 else 10)

        values, c = pydcop.dcop.relations.find_arg_optimal(v1, f1, mode="min")

        self.assertEqual(len(values), 2)
        self.assertIn(4, values)
        self.assertIn(5, values)
        self.assertEqual(c, 2)


################################################################################


def test_find_optimal_single_unary_constraint():
    v1 = Variable("a", [0, 1, 2, 3, 4])
    c1 = UnaryFunctionRelation("c1", v1, lambda x: abs(x - 2))

    vals, cost = find_optimal(v1, {}, [c1], "min")
    assert vals == [2]
    assert cost == 0


def test_find_optimal_2_unary_constraints():
    variable = Variable("a", [0, 1, 2, 3, 4])
    c1 = UnaryFunctionRelation("c1", variable, lambda x: abs(x - 3))
    c2 = UnaryFunctionRelation("c1", variable, lambda x: abs(x - 1) * 2)

    val, sum_costs = find_optimal(variable, {}, [c1, c2], "min")
    assert val == [1]
    assert sum_costs == 2


def test_find_optimal_one_binary_constraint():
    v1 = Variable("v1", [0, 1, 2, 3, 4])
    v2 = Variable("v2", [0, 1, 2, 3, 4])

    @AsNAryFunctionRelation(v1, v2)
    def c1(v1_, v2_):
        return abs(v1_ - v2_)

    val, sum_costs = find_optimal(v1, {"v2": 1}, [c1], "min")
    assert val == [1]
    assert sum_costs == 0


def test_find_optimal_several_best_values():
    #  With this constraints definition, the value 0 and 3 returns the same
    #  optimal cost of 0, find_best_values must return both values.

    v1 = Variable("v1", [0, 1, 2, 3, 4])
    v2 = Variable("v2", [0, 1, 2, 3, 4])

    @AsNAryFunctionRelation(v1, v2)
    def c1(v1_, v2_):
        return abs((v2_ - v1_) % 3)

    val, sum_costs = find_optimal(v1, {"v2": 3}, [c1], "min")
    assert val == [0, 3]
    assert sum_costs == 0


################################################################################


def test_optimal_cost_value():

    x1 = VariableWithCostFunc("x1", list(range(10)), lambda x: x * 2 + 1)

    assert (0, 1) == optimal_cost_value(x1, "min")

    assert (9, 19) == optimal_cost_value(x1, "max")


################################################################################


def test_check_type_by_string():
    pydcop.algorithms.is_of_type_by_str(2, "int")
    pydcop.algorithms.is_of_type_by_str(2.5, "float")
    pydcop.algorithms.is_of_type_by_str("foo", "str")


def test_check_type_by_string_invalid_type():
    assert not pydcop.algorithms.is_of_type_by_str(2, "str")
    assert not pydcop.algorithms.is_of_type_by_str("2.5", "float")
    assert not pydcop.algorithms.is_of_type_by_str(0.25, "int")


def test_is_valid_param_value():
    param_def = pydcop.algorithms.AlgoParameterDef("p", "str", ["a", "b"], "b")

    assert "b" == pydcop.algorithms.check_param_value("b", param_def)
    assert "a" == pydcop.algorithms.check_param_value("a", param_def)

    with pytest.raises(ValueError):
        pydcop.algorithms.check_param_value("invalid_value", param_def)


################################################################################


@pytest.fixture
def algo_param_defs():
    yield [
        pydcop.algorithms.AlgoParameterDef("param1", "str", ["val1", "val2"], "val1"),
        pydcop.algorithms.AlgoParameterDef("param2", "int", None, None),
    ]


def test_algo_parameters_all_defaults(algo_param_defs):
    default_params = pydcop.algorithms.prepare_algo_params({}, algo_param_defs)
    assert "param1" in default_params
    assert default_params["param1"] == "val1"

    assert "param2" in default_params
    assert default_params["param2"] == None


def test_algo_parameters_with_valid_str_param(algo_param_defs):
    params = pydcop.algorithms.prepare_algo_params({"param1": "val2"}, algo_param_defs)
    assert "param1" in params
    assert params["param1"] == "val2"


def test_algo_parameters_with_valid_int_param(algo_param_defs):

    params = pydcop.algorithms.prepare_algo_params({"param2": 5}, algo_param_defs)
    assert "param2" in params
    assert params["param2"] == 5


def test_algo_parameters_with_all_params(algo_param_defs):
    params = pydcop.algorithms.prepare_algo_params(
        {"param1": "val2", "param2": 10}, algo_param_defs
    )
    assert "param1" in params
    assert params["param1"] == "val2"
    assert "param2" in params
    assert params["param2"] == 10


def test_algo_parameters_with_int_conversion(algo_param_defs):
    params = pydcop.algorithms.prepare_algo_params({"param2": "10"}, algo_param_defs)

    assert params["param2"] == 10


def test_algo_parameters_with_invalid_param(algo_param_defs):
    with pytest.raises(ValueError):
        pydcop.algorithms.prepare_algo_params({"invalid_param": "foo"}, algo_param_defs)


def test_algo_parameters_with_invalid_value(algo_param_defs):
    with pytest.raises(ValueError):
        pydcop.algorithms.prepare_algo_params(
            {"break_mode": "invalid"}, algo_param_defs
        )
