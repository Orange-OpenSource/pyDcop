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
from unittest.mock import MagicMock

from pydcop.dcop.objects import (
    VariableDomain,
    ExternalVariable,
    VariableWithCostFunc,
    Variable,
    VariableWithCostDict,
    VariableNoisyCostFunc,
    create_binary_variables,
    BinaryVariable,
    Domain,
    create_variables,
)
from pydcop.utils.expressionfunction import ExpressionFunction
from pydcop.utils.simple_repr import simple_repr, from_repr, SimpleReprException


class TestVariableDomain(unittest.TestCase):
    def test_simple_repr(self):
        d = Domain("d", "foo", [1, 2, 3])
        r = simple_repr(d)
        print(r)

        self.assertEqual(r["__qualname__"], "Domain")
        self.assertEqual(r["__module__"], "pydcop.dcop.objects")
        self.assertEqual(r["name"], "d")
        self.assertEqual(r["domain_type"], "foo")

    def test_from_simple_repr(self):
        d = Domain("d", "foo", [1, 2, 3])
        r = simple_repr(d)
        d2 = from_repr(r)

        self.assertEqual(d, d2)

    def test_hash(self):
        d1 = Domain("d", "foo", [1, 2, 3])
        h1 = hash(d1)
        self.assertIsNotNone(h1)

        d2 = Domain("d", "foo", [1, 2, 4])
        h2 = hash(d2)
        self.assertIsNotNone(h2)

        self.assertNotEqual(h1, h2)

        d3 = Domain("d", "foo", [1, 2, 3])
        h3 = hash(d3)

        self.assertEqual(h1, h3)


class TestVariable(unittest.TestCase):
    def test_list_domain(self):
        v = Variable("v", [1, 2, 3, 4])
        self.assertTrue(v.domain, VariableDomain)

    def test_raises_when_no_domain(self):
        self.assertRaises(ValueError, Variable, "v", None)

    def test_no_initial_value(self):

        v = Variable("v", [1, 2, 3, 4])
        self.assertEqual(v.initial_value, None)

    def test_initial_value(self):

        v = Variable("v", [1, 2, 3, 4], 2)
        self.assertEqual(v.initial_value, 2)

    def test_invalid_initial_value(self):

        self.assertRaises(ValueError, Variable, "v", [1, 2, 3, 4], "A")
        self.assertRaises(ValueError, Variable, "v", [1, 2, 3, 4], initial_value="A")

    def test_simple_repr(self):
        d = VariableDomain("d", "foo", [1, 2, 3])
        v = Variable("v1", d, 2)

        r = simple_repr(v)
        self.assertEqual(r["name"], "v1")
        self.assertEqual(r["initial_value"], 2)
        self.assertEqual(r["domain"], simple_repr(d))

    def test_simple_repr_no_initial_value(self):
        d = VariableDomain("d", "foo", [1, 2, 3])
        v = Variable("v1", d)

        r = simple_repr(v)
        self.assertEqual(r["name"], "v1")
        self.assertEqual(r["initial_value"], None)

        self.assertEqual(r["domain"], simple_repr(d))

    def test_simple_repr_list_based_domain(self):
        v = Variable("v1", [1, 2, 3])

        r = simple_repr(v)
        self.assertEqual(r["name"], "v1")
        self.assertEqual(r["initial_value"], None)

        # domain values are serialized as tuple
        self.assertIn(1, r["domain"]["values"])

    def test_from_simple_repr(self):
        d = VariableDomain("d", "foo", [1, 2, 3])
        v = Variable("v1", d)
        r = simple_repr(v)

        self.assertEqual(v, from_repr(r))

    def test_hash(self):
        d1 = VariableDomain("d", "foo", [1, 2, 3])
        v1 = Variable("v1", d1)
        v2 = Variable("v2", d1)

        self.assertNotEqual(hash(v1), hash(v2))
        self.assertEqual(hash(v1), hash(Variable("v1", d1)))

    def test_hash_diferent_initial_value(self):
        d1 = VariableDomain("d", "foo", [1, 2, 3])
        v1 = Variable("v1", d1, 1)
        vo = Variable("v1", d1, 2)

        self.assertNotEqual(hash(v1), hash(vo))


class TestVariableWithCostDict(unittest.TestCase):
    def test_cost_dict(self):
        v1 = VariableWithCostDict("v1", [1, 2, 3], {1: 0.5, 2: 0.8, 3: 1})

        self.assertEqual(v1.cost_for_val(3), 1)

    def test_cost_dict_simple_repr(self):
        v1 = VariableWithCostDict("v1", [1, 2, 3], {1: 0.5, 2: 0.8, 3: 1})
        r = simple_repr(v1)

        self.assertEqual(r["costs"], {1: 0.5, 2: 0.8, 3: 1})

    def test_cost_dict_from_repr(self):
        v1 = VariableWithCostDict("v1", [1, 2, 3], {1: 0.5, 2: 0.8, 3: 1})
        r = simple_repr(v1)
        v2 = from_repr(r)

        self.assertEqual(v1, v2)

    def test_hash(self):
        d1 = VariableDomain("d", "foo", [1, 2, 3])
        v1 = VariableWithCostDict("v1", d1, {1: 1, 2: 2, 3: 3})
        v1_othercosts = VariableWithCostDict("v1", d1, {1: 2, 2: 2, 3: 3})

        self.assertNotEqual(hash(v1), hash(v1_othercosts))

        v1_othername = VariableWithCostDict("v1_other", d1, {1: 1, 2: 2, 3: 3})

        self.assertNotEqual(hash(v1), hash(v1_othername))

        self.assertEqual(
            hash(v1), hash(VariableWithCostDict("v1", d1, {1: 1, 2: 2, 3: 3}))
        )


class TestVariableWithCostFunc(unittest.TestCase):
    def test_with_lambda_cost_func(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = VariableWithCostFunc("v", domain, cost_func=lambda x: x * 2)

        self.assertEqual(v.cost_for_val(1), 2)
        self.assertEqual(v.cost_for_val(3), 6)
        self.assertEqual(v.cost_for_val(4), 8)

    def test_with_named_cost_func(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])

        def cost_func(val):
            return val / 2

        v = VariableWithCostFunc("v", domain, cost_func=cost_func)

        self.assertEqual(v.cost_for_val(1), 0.5)
        self.assertEqual(v.cost_for_val(3), 1.5)
        self.assertEqual(v.cost_for_val(4), 2)

    def test_with_expression_based_cost_func(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v +1")
        v = VariableWithCostFunc("v", domain, cost_func=cost_func)

        self.assertEqual(v.cost_for_val(1), 2)
        self.assertEqual(v.cost_for_val(3), 4)

    def test_raise_on_expression_with_wrong_variable(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("w +1")
        self.assertRaises(
            ValueError, VariableWithCostFunc, "v", domain, cost_func=cost_func
        )

    def test_raise_on_expression_with_many_variable(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v+ w +1")
        self.assertRaises(
            ValueError, VariableWithCostFunc, "v", domain, cost_func=cost_func
        )

    def test_simple_repr_not_supported_with_arbitrary_cost_func(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])

        def cost_func(val):
            return val / 2

        v = VariableWithCostFunc("v", domain, cost_func=cost_func)

        self.assertRaises(SimpleReprException, simple_repr, v)

    def test_simple_repr_with_expression_function(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v / 2")
        v = VariableWithCostFunc("v", domain, cost_func=cost_func)

        r = simple_repr(v)

        self.assertEqual(r["name"], "v")
        self.assertEqual(r["cost_func"]["expression"], "v / 2")

    def test_from_repr_with_expression_function(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v / 2")
        v = VariableWithCostFunc("v", domain, cost_func=cost_func)

        r = simple_repr(v)
        v2 = from_repr(r)

        self.assertEqual(v2.cost_for_val(2), v.cost_for_val(2))
        self.assertEqual(v2, v)

    def test_hash(self):
        d1 = VariableDomain("d", "foo", [1, 2, 3])
        v1 = VariableWithCostFunc("v1", d1, cost_func=ExpressionFunction("v1/2"))
        v1_othercosts = VariableWithCostFunc(
            "v1", d1, cost_func=ExpressionFunction("v1/3")
        )

        self.assertNotEqual(hash(v1), hash(v1_othercosts))

        v1_othername = VariableWithCostFunc(
            "v1_other", d1, cost_func=ExpressionFunction("v1_other/2")
        )

        self.assertNotEqual(hash(v1), hash(v1_othername))

        self.assertEqual(
            hash(v1),
            hash(VariableWithCostFunc("v1", d1, cost_func=ExpressionFunction("v1/2"))),
        )


class TestVariableWithNoisyFunctionCost(unittest.TestCase):
    def test_simple_repr_with_expression_function(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v / 2")
        v = VariableNoisyCostFunc("v", domain, cost_func=cost_func)

        r = simple_repr(v)

        self.assertEqual(r["name"], "v")
        self.assertEqual(r["cost_func"]["expression"], "v / 2")

    def test_from_repr_with_expression_function(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        cost_func = ExpressionFunction("v / 2")
        v = VariableNoisyCostFunc("v", domain, cost_func=cost_func)

        r = simple_repr(v)
        v2 = from_repr(r)

        # Due to the noise, the cost will de different
        c1 = v2.cost_for_val(2)
        c2 = v.cost_for_val(2)
        self.assertLessEqual(abs(c1 - c2), v.noise_level * 2)

        self.assertEqual(v2, v)

    def test_hash(self):
        d1 = VariableDomain("d", "foo", [1, 2, 3])
        v1 = VariableNoisyCostFunc("v1", d1, cost_func=ExpressionFunction("v1/2"))
        v1_othercosts = VariableNoisyCostFunc(
            "v1", d1, cost_func=ExpressionFunction("v1/3")
        )

        self.assertNotEqual(hash(v1), hash(v1_othercosts))

        v1_othername = VariableNoisyCostFunc(
            "v1_other", d1, cost_func=ExpressionFunction("v1_other/2")
        )

        self.assertNotEqual(hash(v1), hash(v1_othername))

        self.assertEqual(
            hash(v1),
            hash(VariableNoisyCostFunc("v1", d1, cost_func=ExpressionFunction("v1/2"))),
        )


class TestExternalVariables(unittest.TestCase):
    def test_create(self):

        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)

        self.assertEqual(v.name, "v")
        self.assertEqual(v.domain, domain)
        self.assertEqual(v.value, 1)

    def test_clone(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)
        v_clone = v.clone()
        self.assertEqual(v_clone.name, "v")
        self.assertEqual(v_clone.domain, domain)
        self.assertEqual(v_clone.value, 1)

    def test_create_no_initial_value(self):

        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        # As None is not in the domain, the initial value MUST be given when
        # creating the variable
        self.assertRaises(ValueError, ExternalVariable, "v", domain)

    def test_set_value_in_domain(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)

        v.value = 2
        self.assertEqual(v.value, 2)

    def test_set_value_not_in_domain(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)

        def set_value(value):
            v.value = value

        # 5 is not in the domain, this must raises and ValueError
        self.assertRaises(ValueError, set_value, 5)
        self.assertEqual(v.value, 1)

    def test_cb_subscribed(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)
        cb = MagicMock()

        v.subscribe(cb)
        v.value = 2
        cb.assert_called_with(2)

    def test_cb_subscribed_no_change(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)
        cb = MagicMock()
        v.value = 2
        v.subscribe(cb)

        # The callback must be called only if the value has really changed
        v.value = 2
        self.assertFalse(cb.called)
        v.value = 3
        cb.assert_called_with(3)
        cb.reset_mock()

        v.value = 3
        self.assertFalse(cb.called)

    def test_cb_unsusbcribed(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)
        cb = MagicMock()

        v.subscribe(cb)
        v.unsubscribe(cb)
        v.value = 2
        self.assertFalse(cb.called)

    def test_clone_cb(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)
        cb = MagicMock()
        cb_clone = MagicMock()

        v.subscribe(cb)
        v_clone = v.clone()
        v_clone.subscribe(cb_clone)
        v_clone.value = 3

        self.assertFalse(cb.called)
        cb_clone.assert_called_with(3)

    def test_simple_repr(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)

        r = simple_repr(v)

        self.assertEqual(r["name"], "v")
        self.assertEqual(r["value"], 1)

    def test_from_repr(self):
        domain = VariableDomain("d", "d", [1, 2, 3, 4])
        v = ExternalVariable("v", domain, value=1)

        r = simple_repr(v)
        v1 = from_repr(r)
        self.assertEqual(v, v1)


class TestMassVariableCreation(unittest.TestCase):
    def test_create_several_variables_from_list(self):
        d = Domain("color", "", ["R", "G", "B"])
        variables = create_variables("x_", ["a1", "a2", "a3"], d)

        self.assertIn("x_a1", variables)
        self.assertTrue(isinstance(variables["x_a2"], Variable))
        self.assertEqual(variables["x_a3"].name, "x_a3")

    def test_create_several_variables_from_range(self):
        d = Domain("color", "", ["R", "G", "B"])
        variables = create_variables("x_", range(10), d)

        self.assertIn("x_1", variables)
        self.assertTrue(isinstance(variables["x_2"], Variable))
        self.assertEqual(variables["x_3"].name, "x_3")

    def test_create_several_variables_from_several_lists(self):
        d = Domain("color", "", ["R", "G", "B"])
        variables = create_variables("m_", (["x1", "x2"], ["a1", "a2", "a3"]), d)

        self.assertEqual(len(variables), 6)
        self.assertIn(("x1", "a2"), variables)
        self.assertTrue(isinstance(variables[("x2", "a3")], Variable))
        self.assertEqual(variables[("x2", "a3")].name, "m_x2_a3")

    def test_create_several_binvariables_from_list(self):
        variables = create_binary_variables("x_", ["a1", "a2", "a3"])

        self.assertIn("x_a1", variables)
        self.assertTrue(isinstance(variables["x_a2"], BinaryVariable))
        self.assertEqual(variables["x_a3"].name, "x_a3")

    def test_create_several_binvariables_from_several_lists(self):
        variables = create_binary_variables("m_", (["x1", "x2"], ["a1", "a2", "a3"]))

        self.assertEqual(len(variables), 6)
        self.assertIn(("x1", "a2"), variables)
        self.assertTrue(isinstance(variables[("x2", "a3")], BinaryVariable))
        self.assertEqual(variables[("x2", "a3")].name, "m_x2_a3")
