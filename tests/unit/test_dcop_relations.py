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

import pydcop.dcop.objects
from pydcop.dcop.objects import (
    VariableDomain,
    Variable,
    ExternalVariable,
    Domain,
    VariableWithCostFunc,
)
from pydcop.dcop.relations import (
    NAryFunctionRelation,
    is_compatible,
    ConditionalRelation,
    count_var_match,
    AsNAryFunctionRelation,
    relation_from_str,
    find_dependent_relations,
    NAryMatrixRelation,
    UnaryBooleanRelation,
    UnaryFunctionRelation,
    ZeroAryRelation,
    add_var_to_rel,
    NeutralRelation,
    assignment_matrix,
    random_assignment_matrix,
    generate_assignment_as_dict,
    constraint_from_str,
    assignment_cost,
)
from pydcop.utils.expressionfunction import ExpressionFunction
from pydcop.utils.simple_repr import simple_repr, from_repr, SimpleReprException


class ZeroAryRelationTest(unittest.TestCase):
    def setUp(self):
        self.r0 = ZeroAryRelation("r0", 42)

    def test_properties(self):
        self.assertEqual(self.r0.arity, 0)
        self.assertEqual(self.r0.name, "r0")
        self.assertEqual(self.r0.dimensions, [])
        self.assertEqual(self.r0.shape, ())

    def test_get_value(self):

        self.assertEqual(self.r0(), 42)

        # should raise an exception
        self.assertRaises(ValueError, self.r0, ["x1"], ["a"])

        self.assertEqual(self.r0.get_value_for_assignment([]), 42)

        # should raise an exception
        self.assertRaises(ValueError, self.r0.get_value_for_assignment, ["x1"])

    def test_set_value(self):

        r1 = self.r0.set_value_for_assignment({}, 21)
        self.assertEqual(r1(), 21)

    def test_slicing_on_no_var_is_ok(self):
        r1 = self.r0.slice({})
        self.assertEqual(r1(), 42)

    def test_slicing_on_variable_raises_valueerror(self):

        # should raise an exception when slicing with args
        self.assertRaises(ValueError, self.r0.slice, {"x1": "a"})

    def test_set_value_for_assignement(self):

        r1 = self.r0.set_value_for_assignment({}, 4)

        self.assertEqual(r1(), 4)
        self.assertNotEqual(id(self.r0), id(r1))
        self.assertNotEqual(hash(self.r0), hash(r1))

    def test_simple_repr(self):
        r = simple_repr(self.r0)

        self.assertEqual(r["name"], "r0")
        self.assertEqual(r["value"], 42)

    def test_from_repr(self):
        r = simple_repr(self.r0)

        r1 = from_repr(r)

        self.assertEqual(r1, self.r0)

    def test_hash(self):
        h = hash(self.r0)
        self.assertIsNotNone(h)

        self.assertEqual(h, hash(ZeroAryRelation("r0", 42)))
        self.assertNotEqual(h, hash(ZeroAryRelation("r1", 42)))
        self.assertNotEqual(h, hash(ZeroAryRelation("r0", 43)))


class UnaryFunctionRelationTest(unittest.TestCase):
    def setUp(self):
        self.x1 = Variable("x1", [1, 3, 5])
        self.r0 = UnaryFunctionRelation("r0", self.x1, lambda x: x * 2)

    def test_properties(self):
        self.assertEqual(self.r0.arity, 1)
        self.assertEqual(self.r0.name, "r0")
        self.assertSequenceEqual(self.r0.dimensions, [self.x1])
        self.assertEqual(self.r0.shape, (3,))

    def test_slicing_on_no_var_is_ok(self):

        r1 = self.r0.slice({})

        self.assertEqual(r1.arity, 1)
        self.assertEqual(r1(3), 6)

    def test_slicing_on_existing_var_is_ok(self):

        r1 = self.r0.slice({self.x1.name: 3})

        self.assertEqual(r1.arity, 0)
        self.assertEqual(r1(), 6)

    def test_slicing_on_unknown_var_raises_valueerror(self):

        self.assertRaises(ValueError, self.r0.slice, {"unknown": 3})

    def test_slicing_on_more_than_one_var_raises_valueerror(self):

        self.assertRaises(ValueError, self.r0.slice, {"x1": 3, "x2": 5})
        self.assertRaises(ValueError, self.r0.slice, {"x1": 3, "x2": 5, "x3": 7})

    def test_get_value(self):

        self.assertEqual(self.r0(3), 6)
        self.assertRaises(ValueError, self.r0, 3, 4)

    def test_get_value_dict(self):

        self.assertEqual(self.r0(**{"x1": 3}), 6)

    def test_eq(self):
        f1 = UnaryFunctionRelation("f1", self.x1, ExpressionFunction("x1 * 0.8"))
        f2 = UnaryFunctionRelation("f1", self.x1, ExpressionFunction("x1 * 0.8"))

        self.assertEqual(f1, f2)

    def test_not_eq(self):
        f1 = UnaryFunctionRelation("f1", self.x1, ExpressionFunction("x1 * 0.8"))
        f2 = UnaryFunctionRelation("f2", self.x1, ExpressionFunction("x1 * 0.8"))

        self.assertNotEqual(f1, f2)

    def test_raise_on_simple_repr_witharbitrary_function(self):
        self.assertRaises(SimpleReprException, simple_repr, self.r0)

    def test_simple_repr_with_expression_function(self):
        f1 = UnaryFunctionRelation("f1", self.x1, ExpressionFunction("x1 * 0.8"))

        r = simple_repr(f1)
        self.assertEqual(r["name"], "f1")
        self.assertEqual(r["variable"]["name"], "x1")
        self.assertEqual(r["rel_function"]["expression"], "x1 * 0.8")

    def test_from_repr_with_expression_function(self):
        f1 = UnaryFunctionRelation("f1", self.x1, ExpressionFunction("x1 * 0.8"))

        r = simple_repr(f1)
        f2 = from_repr(r)

        self.assertEqual(f2.name, "f1")
        self.assertSequenceEqual(f2.dimensions, [self.x1])
        self.assertEqual(f1.expression, f2.expression)
        self.assertEqual(f1, f2)

    def test_hash(self):
        h = hash(self.r0)
        self.assertIsNotNone(h)

        # When using lambda, the relation have different hash
        self.assertNotEqual(
            h, hash(UnaryFunctionRelation("r0", self.x1, lambda x: x * 2))
        )

    def test_hash_expression(self):
        r0 = UnaryFunctionRelation("r0", self.x1, ExpressionFunction("x * 2"))
        h = hash(r0)
        self.assertIsNotNone(h)

        self.assertEqual(
            h, hash(UnaryFunctionRelation("r0", self.x1, ExpressionFunction("x * 2")))
        )

        self.assertNotEqual(
            h, hash(UnaryFunctionRelation("r1", self.x1, ExpressionFunction("x * 2")))
        )

        x2 = Variable("x2", [1, 3, 5])
        self.assertNotEqual(
            h, hash(UnaryFunctionRelation("r0", x2, ExpressionFunction("x * 2")))
        )


class UnaryBooleanRelationTest(unittest.TestCase):
    def setUp(self):
        self.x1 = Variable("x1", [0, 3, False, True])
        self.r0 = UnaryBooleanRelation("r0", self.x1)

    def test_properties(self):
        self.assertEqual(self.r0.arity, 1)
        self.assertEqual(self.r0.name, "r0")
        self.assertEqual(self.r0.dimensions, [self.x1])
        self.assertEqual(self.r0.shape, (4,))

    def test_slice_on_known_var_is_ok(self):
        r1 = self.r0.slice({self.x1.name: 3})

        self.assertEqual(r1.arity, 0)
        self.assertEqual(r1(), True)

        r1 = self.r0.slice({self.x1.name: 0})
        self.assertEqual(r1.arity, 0)
        self.assertEqual(r1(), False)

        r1 = self.r0.slice({self.x1.name: False})
        self.assertEqual(r1.arity, 0)
        self.assertEqual(r1(), False)

        r1 = self.r0.slice({self.x1.name: True})
        self.assertEqual(r1.arity, 0)
        self.assertEqual(r1(), True)

    def test_slice_on_unknown_var_raises(self):

        self.assertRaises(ValueError, self.r0.slice, {"unknown": 4})

    def test_slice_on_more_than_one_var_raises(self):

        self.assertRaises(ValueError, self.r0.slice, {"u1": 4, "u2": 8})

    def test_get_value(self):
        self.assertEqual(self.r0(3), True)
        self.assertEqual(self.r0(True), True)
        self.assertEqual(self.r0(0), False)
        self.assertEqual(self.r0(False), False)
        self.assertRaises(ValueError, self.r0, 3, 4)

        self.assertEqual(self.r0.get_value_for_assignment([0]), False)
        self.assertEqual(self.r0.get_value_for_assignment({"x1": 3}), True)

    def test_simple_repr(self):
        r = simple_repr(self.r0)

        self.assertEqual(r["name"], "r0")
        self.assertEqual(r["var"]["name"], "x1")

    def test_from_repr(self):
        r = simple_repr(self.r0)
        r1 = from_repr(r)

        self.assertEqual(r1, self.r0)
        self.assertEqual(r1(3), True)

    def test_hash(self):
        h = hash(self.r0)
        self.assertIsNotNone(h)

        self.assertEqual(h, hash(UnaryBooleanRelation("r0", self.x1)))

        self.assertNotEqual(h, hash(UnaryBooleanRelation("r1", self.x1)))

        x2 = Variable("x2", [1, 3, 5])
        self.assertNotEqual(h, UnaryBooleanRelation("r0", x2))


class NAryFunctionRelationTests(unittest.TestCase):
    def init_with_name(self):
        x1 = Variable("x1", [2, 4, 1])

        def f(x):
            return x * 3

        r1 = NAryFunctionRelation(f, [x1], name="pouet")

        self.assertEqual(r1.name, "pouet")

    def test_init_one_var(self):
        x1 = Variable("x1", [2, 4, 1])

        def f(x):
            return x * 3

        r1 = NAryFunctionRelation(f, [x1])

        self.assertEqual(r1.name, "f")
        self.assertEqual(r1.arity, 1)
        self.assertEqual(r1.dimensions, [x1])
        self.assertEqual(r1.shape, (3,))

    def test_slice_with_no_var_is_ok(self):
        x1 = Variable("x1", [2, 4, 1])
        x2 = Variable("x2", [2, 4, 1])

        def f(x, y):
            return x * 3 + y

        r1 = NAryFunctionRelation(f, [x1, x2])

        r2 = r1.slice({})

        self.assertEqual(r2.arity, 2)
        self.assertEqual(r2.dimensions, [x1, x2])
        self.assertEqual(r2(x1=4, x2=3), 15)

    def test_slice_with_too_many_var_raises(self):
        x1 = Variable("x1", [2, 4, 1])

        def f(x):
            return x * 3

        r1 = NAryFunctionRelation(f, [x1])
        self.assertRaises(ValueError, r1.slice, {x1.name: 4, "invalid": 8})

    def test_slice_on_unknown_var_raises(self):
        x1 = Variable("x1", [2, 4, 1])
        x2 = Variable("x2", [2, 4, 1])

        def f(x, y):
            return x * 3 + y

        r1 = NAryFunctionRelation(f, [x1, x2])
        self.assertRaises(ValueError, r1.slice, {"invalid": 8})
        self.assertRaises(ValueError, r1.slice, {x1.name: 4, "invalid": 8})

    def test_slice_one_var_is_ok(self):
        x1 = Variable("x1", [2, 4, 1])

        def f(x):
            return x * 3

        r1 = NAryFunctionRelation(f, [x1])

        r2 = r1.slice({x1.name: 4})

        self.assertEqual(r2.arity, 0)
        self.assertEqual(r2.dimensions, [])
        self.assertEqual(r2(), 12)

    def test_slice_two_var_is_ok(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2])

        r2 = r1.slice({x1.name: 4})

        self.assertEqual(r2.arity, 1)
        self.assertEqual(r2.dimensions, [x2])
        self.assertEqual(r2(5), 9)

    def test_get_value_as_array(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2])

        # getting value with get_value_for_assignment
        self.assertEqual(r1.get_value_for_assignment([2, 3]), 5)

        # getting value with callable
        self.assertEqual(r1(2, 3), 5)

    def test_get_value_as_dict(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        def f(x, y):
            return x - y

        r1 = NAryFunctionRelation(f, [x1, x2])

        # getting value with get_value_for_assignment with a dict
        # using a dict avoid getting the order of the variable wrong
        self.assertEqual(r1.get_value_for_assignment({"x2": 1, "x1": 4}), 3)
        self.assertEqual(r1.get_value_for_assignment({"x1": 1, "x2": 4}), -3)

    def test_fct_1vars(self):
        # A Factor with only one variable !

        d1 = ["a", "b", "c"]
        x1 = Variable("x1", d1)

        def fct(x):
            return 1 if x == "a" else 0

        ff = NAryFunctionRelation(fct, [x1])

        self.assertEqual(ff("a"), 1)
        self.assertEqual(ff("b"), 0)
        self.assertEqual(ff("c"), 0)

    def test_fct_2vars(self):
        d1 = ["a", "b", "c"]
        d2 = ["a", "b"]

        x1 = Variable("x1", d1)
        x2 = Variable("x2", d2)

        def fct(a1, a2):
            return 1 if a1 == a2 else 0

        ff = NAryFunctionRelation(fct, [x1, x2])

        self.assertEqual(ff("a", "b"), 0)
        self.assertEqual(ff("a", "a"), 1)
        self.assertEqual(ff("b", "b"), 1)
        self.assertEqual(ff.get_value_for_assignment(["a", "b"]), 0)
        self.assertEqual(ff.get_value_for_assignment(["b", "b"]), 1)
        self.assertEqual(ff.get_value_for_assignment(["c", "b"]), 0)

    def test_fct_3vars(self):
        d1 = ["a", "b", "c"]
        d2 = ["a", "b"]
        d3 = ["a", "b", "c"]

        x1 = Variable("x1", d1)
        x2 = Variable("x2", d2)
        x3 = Variable("x3", d3)

        def fct(x1_, x2_, x3_):
            c1 = 1 if x1_ == x2_ else 0
            c2 = 1 if x1_ == x3_ else 0
            c3 = 1 if x2_ == x3_ else 0
            return c1 + c2 + c3

        ff = NAryFunctionRelation(fct, [x1, x2, x3])

        self.assertEqual(ff("a", "b", "c"), 0)
        self.assertEqual(ff.get_value_for_assignment(["a", "b", "c"]), 0)
        self.assertEqual(
            ff.get_value_for_assignment({"x1": "a", "x2": "b", "x3": "c"}), 0
        )
        self.assertEqual(ff("b", "b", "c"), 1)
        self.assertEqual(ff("b", "b", "b"), 3)

    def test_eq(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2])
        r2 = NAryFunctionRelation(f, [x1, x2])

        self.assertEqual(r1, r2)

    def test_eq_explicit_name(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2], name="f")
        r2 = NAryFunctionRelation(f, [x1, x2], name="f")

        self.assertEqual(r1, r2)

    def test_eq_expression_function(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])

        r1 = NAryFunctionRelation(ExpressionFunction("x1 + x2"), [x1, x2], name="f")
        r2 = NAryFunctionRelation(ExpressionFunction("x1 + x2"), [x1, x2], name="f")

        self.assertEqual(r1, r2)

    def test_not_eq(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2], name="r1")
        r2 = NAryFunctionRelation(f, [x1, x2])
        r3 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name="r1")
        r4 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name="r4")

        self.assertNotEqual(r1, r2)
        self.assertNotEqual(r1, r3)
        self.assertNotEqual(r1, r4)

    def test_raise_on_simple_repr_with_arbitrary_function(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])

        def f(x, y):
            return x + y

        r1 = NAryFunctionRelation(f, [x1, x2], name="r1")
        with self.assertRaises(SimpleReprException):
            simple_repr(r1)

    def test_simple_repr_with_expression_function(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])
        r1 = NAryFunctionRelation(ExpressionFunction("x1 + x2"), [x1, x2], name="r1")

        r = simple_repr(r1)

        print(r)
        self.assertEqual(r["name"], "r1")
        self.assertEqual(len(r["variables"]), 2)
        self.assertEqual(r["variables"][0]["name"], "x1")

    def test_from_repr_with_expression_function(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2])
        r1 = NAryFunctionRelation(ExpressionFunction("x1 + x2"), [x1, x2], name="r1")

        r = simple_repr(r1)
        r2 = from_repr(r)

        self.assertEqual(r1, r2)

    def test_to_repr_when_slicing(self):
        x1 = Variable("x1", [2, 4, 1])
        r1 = NAryFunctionRelation(ExpressionFunction("x1 * 3"), [x1])

        r2 = r1.slice({x1.name: 4})

        self.assertEqual(r2.arity, 0)
        self.assertEqual(r2.dimensions, [])
        self.assertEqual(r2(), 12)

        r = simple_repr(r2)
        self.assertIsNotNone(r)

    def test_to_repr_when_slicing_2(self):
        x1 = Variable("x1", [1, 2, 3])
        x2 = Variable("x2", [1, 2, 3])
        r1 = NAryFunctionRelation(ExpressionFunction("x1 + x2"), [x1, x2], name="r1")

        r2 = r1.slice({x1.name: 4})

        self.assertEqual(r2.arity, 1)
        self.assertEqual(r2.dimensions, [x2])
        self.assertEqual(r2(x2=10), 14)
        self.assertEqual(r1.name, "r1")
        self.assertEqual(r2.name, "r1")

        r = simple_repr(r2)
        self.assertIsNotNone(r)

    def test_hash_with_expressionfunction(self):
        x1 = Variable("x1", [2, 4, 1])
        x2 = Variable("x2", [2, 4, 1])
        r1 = NAryFunctionRelation(ExpressionFunction("x1 * 3 + x2"), [x1, x2])

        h = hash(r1)
        self.assertIsNotNone(h)

        self.assertEqual(
            h, hash(NAryFunctionRelation(ExpressionFunction("x1 * 3 + x2"), [x1, x2]))
        )

        self.assertNotEqual(
            h,
            hash(
                NAryFunctionRelation(
                    ExpressionFunction("x1 * 3 + x2"), [x1, x2], name="foo"
                )
            ),
        )

        self.assertNotEqual(
            h, hash(NAryFunctionRelation(ExpressionFunction("x1 * 2 + x2"), [x1, x2]))
        )

    def test_hash_not_equal_with_lambda(self):
        x1_v = Variable("x1", [2, 4, 1])
        x2_v = Variable("x2", [2, 4, 1])
        r1 = NAryFunctionRelation(lambda x1, x2: x1 * 3 + x2, [x1_v, x2_v])

        h = hash(r1)
        self.assertIsNotNone(h)

        self.assertNotEqual(
            h, hash(NAryFunctionRelation(lambda x1, x2: x1 * 3 + x2, [x1_v, x2_v]))
        )

    def test_function_with_kwargs(self):

        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])

        def f(**kwargs):
            r = 0
            for k in kwargs:
                r += kwargs[k]
            return r

        r = NAryFunctionRelation(f, [v1, v2, v3], "f_rel")

        obtained = r(v1=2, v3=1, v2=1)

        self.assertEqual(obtained, 4)

        sliced = r.slice({"v1": 3})
        self.assertEqual(sliced(v2=2, v3=2), 7)
        self.assertEqual(len(sliced.dimensions), 2)
        self.assertIn(v2, sliced.dimensions)
        self.assertIn(v3, sliced.dimensions)

    def test_function_with_varargs(self):

        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])

        def f(*args):
            c = 0
            for a in args:
                c += args
            return c

        r = NAryFunctionRelation(f, [v1, v2, v3], "f_rel")

        with self.assertRaises(TypeError):
            obtained = r(v1=2, v3=1, v2=1)


class NAryFunctionRelationDecoratorTests(unittest.TestCase):
    def test_1var(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)

        @AsNAryFunctionRelation(x1)
        def x1_cost(x):
            return x * 0.8

        self.assertEqual(x1_cost.name, "x1_cost")
        self.assertEqual(x1_cost.arity, 1)
        self.assertIn("x1", [v.name for v in x1_cost.dimensions])
        self.assertEqual(x1_cost.dimensions, [x1])

        self.assertEqual(x1_cost(2), 1.6)

    def test_2var(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x, y):
            return x + y

        self.assertEqual(phi.name, "phi")
        self.assertEqual(phi.arity, 2)
        self.assertIn("x1", [v.name for v in phi.dimensions])
        self.assertIn("x2", [v.name for v in phi.dimensions])

        self.assertEqual(phi(2, 3), 5)


def get_1var_rel():
    x = Variable("x1", ["a", "b", "c"])
    u = NAryMatrixRelation([x])

    return x, u


def get_2var_rel():
    x1 = Variable("x1", ["a", "b", "c"])
    x2 = Variable("x2", ["1", "2"])
    u1 = NAryMatrixRelation([x1, x2], [[1, 2], [3, 4], [5, 6]])

    return x1, x2, u1


class NAryMatrixRelationInitTest(unittest.TestCase):
    def test_init_zero_no_var(self):
        u1 = NAryMatrixRelation([])

        self.assertEqual(u1.dimensions, [])
        self.assertEqual(u1.arity, 0)

        val = u1.get_value_for_assignment([])
        self.assertEqual(val, 0)

    def test_init_zero_one_var(self):

        x1, u1 = get_1var_rel()

        self.assertEqual(u1.dimensions, [x1])
        self.assertEqual(u1.arity, 1)
        self.assertEqual(u1.shape, (3,))

    def test_init_zero_2var(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation([x1, x2])

        self.assertEqual(u1.dimensions, [x1, x2])
        self.assertEqual(u1.arity, 2)
        self.assertEqual(u1.shape, (3, 2))

    def test_init_matrix_one_var(self):

        x1, u1 = get_1var_rel()

        self.assertEqual(u1.dimensions, [x1])
        self.assertEqual(u1.arity, 1)
        self.assertEqual(u1.shape, (3,))

    def test_init_array_one_var(self):
        x1 = Variable("x1", ["a", "b", "c"])

        u1 = NAryMatrixRelation([x1], [0, 2, 3])

        self.assertEqual(u1.dimensions, [x1])
        self.assertEqual(u1.arity, 1)
        self.assertEqual(u1.shape, (3,))
        self.assertEqual(u1("b"), 2)

    def test_init_nparray_one_var(self):
        x1 = Variable("x1", ["a", "b", "c"])

        u1 = NAryMatrixRelation([x1], np.array([0, 2, 3]))

        self.assertEqual(u1.dimensions, [x1])
        self.assertEqual(u1.arity, 1)
        self.assertEqual(u1.shape, (3,))
        self.assertEqual(u1("b"), 2)

    def test_init_matrix_three_var(self):

        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        x3 = Variable("x3", ["y", "z"])

        matrix = [  # for x1 = a
            [
                [1, 2],  # values when x2=1, x3 = y or z
                [3, 4],
            ],  # values when x2=2, x3 = y or z
            # for x1 = b
            [[5, 6], [7, 8]],  # values when
            # for x1 = c
            [[9, 10], [11, 12]],
        ]
        u1 = NAryMatrixRelation([x1, x2, x3], np.array(matrix))

        self.assertEqual(u1.get_value_for_assignment(["a", "2", "z"]), 4)
        self.assertEqual(u1.get_value_for_assignment(["b", "1", "y"]), 5)
        self.assertEqual(u1.get_value_for_assignment(["c", "1", "z"]), 10)

    def test_value_one_var(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1])
        print(u1.get_value_for_assignment(["a"]))

    def test_init_matrix_2var(self):
        x1, x2, u1 = get_2var_rel()

        self.assertEqual(u1.dimensions, [x1, x2])
        self.assertEqual(u1.arity, 2)
        self.assertEqual(u1.shape, (3, 2))

    def test_init_from_generated_matrix(self):
        d = Domain("d", "d", range(10))
        v1 = Variable("v1", d)
        v2 = Variable("v2", d)
        matrix = assignment_matrix([v1, v2], 0)
        r = NAryMatrixRelation([v1, v2], matrix, "r")

        obtained = r(v1=1, v2=0)
        self.assertEqual(obtained, 0)

    def test_value_matrix_one_var(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))

        self.assertEqual(u1.get_value_for_assignment(["b"]), 2)
        self.assertEqual(u1.get_value_for_assignment(["a"]), 1)
        self.assertEqual(u1.get_value_for_assignment(["c"]), 3)

    def test_value_matrix_2var(self):
        x1, x2, u1 = get_2var_rel()

        self.assertEqual(u1.get_value_for_assignment(["b", "2"]), 4)
        self.assertEqual(u1.get_value_for_assignment(["c", "1"]), 5)

    def test_get_value_as_array(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        u1 = NAryMatrixRelation([x1, x2], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # When getting value with get_value_for_assignment with a list,
        # we must make sure the values are in the right order.
        self.assertEqual(u1.get_value_for_assignment([4, 1]), 4)
        self.assertEqual(u1.get_value_for_assignment([2, 5]), 3)

    def test_get_value_as_dict(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        u1 = NAryMatrixRelation([x1, x2], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Getting value with get_value_for_assignment with a dict
        # using a dict avoid getting the order of the variable wrong.
        self.assertEqual(u1.get_value_for_assignment({"x2": 1, "x1": 4}), 4)
        self.assertEqual(u1.get_value_for_assignment({"x1": 2, "x2": 5}), 3)

        # as a callable
        self.assertEqual(u1(x2=1, x1=4), 4)
        self.assertEqual(u1(x1=2, x2=5), 3)

    def test_set_value_as_array(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        u1 = NAryMatrixRelation([x1, x2], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        u2 = u1.set_value_for_assignment([4, 1], 0)

        # When setting value with get_value_for_assignment with a list,
        # we must make sure the values are in the right order.
        self.assertEqual(u2.get_value_for_assignment([4, 1]), 0)

        u2 = u1.set_value_for_assignment([2, 5], 0)
        self.assertEqual(u2.get_value_for_assignment([2, 5]), 0)

    def test_set_value_as_dict(self):
        x1 = Variable("x1", [2, 4, 6])
        x2 = Variable("x2", [1, 3, 5])

        u1 = NAryMatrixRelation([x1, x2], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        u2 = u1.set_value_for_assignment({"x2": 1, "x1": 4}, 0)

        self.assertEqual(u2.get_value_for_assignment([4, 1]), 0)
        self.assertEqual(u2.get_value_for_assignment({"x2": 1, "x1": 4}), 0)

        u2 = u1.set_value_for_assignment({"x1": 2, "x2": 5}, 0)
        self.assertEqual(u2.get_value_for_assignment([2, 5]), 0)

        u2 = u1.set_value_for_assignment({"x1": 2, "x2": 5}, 3)
        self.assertEqual(u2.get_value_for_assignment([2, 5]), 3)

    def test_set_float_value_on_zeroed_init(self):
        x1 = Variable("x1", ["R", "G"])
        x2 = Variable("x2", ["R", "G"])
        u1 = NAryMatrixRelation([x1, x2], name="u1")

        # we set a float vlaue, we want to find our float back !
        u1 = u1.set_value_for_assignment({"x1": "R", "x2": "G"}, 5.2)
        self.assertEqual(u1(**{"x1": "R", "x2": "G"}), 5.2)


class NAryMatrixRelationSliceTest(unittest.TestCase):
    def test_slice_1var(self):

        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))

        # Slicing a unary relation on its only variable gives us a 0 ary
        # relation :
        s = u1.slice({x1.name: "a"})
        self.assertEqual(s.arity, 0)
        self.assertEqual(s.get_value_for_assignment(), 1)

    def test_slice_2var(self):
        x1, x2, u1 = get_2var_rel()

        s = u1.slice({x1.name: "a"})
        self.assertEqual(s.arity, 1)
        self.assertEqual(s.shape, (len(x2.domain),))
        self.assertEqual(s.get_value_for_assignment(["2"]), 2)

    def test_slice_2var_ignore_extra(self):
        x1, x2, u1 = get_2var_rel()

        # When setting ignore_extra_vars, it must silently ignore the extra
        # x4 variable
        x4 = Variable("x4", [1, 2])

        s = u1.slice({x1.name: "a", x4.name: 1}, ignore_extra_vars=True)
        self.assertEqual(s.arity, 1)
        self.assertEqual(s.shape, (len(x2.domain),))
        self.assertEqual(s.get_value_for_assignment(["2"]), 2)


class NAryMatrixRelationFromFunctionTests(unittest.TestCase):
    def test_constant_relation(self):

        f = relation_from_str("f", "4", [])
        rel = NAryMatrixRelation.from_func_relation(f)

        self.assertEqual(type(rel), NAryMatrixRelation)
        self.assertEqual(rel.dimensions, [])
        self.assertEqual(rel(), f())

    def test_3vars_relation(self):

        x1 = Variable("x1", [1, 2, 3, 4])
        x2 = Variable("x2", [1, 2, 3, 4])
        x3 = Variable("x3", [1, 2, 3, 4])
        f = relation_from_str("f", "x1 + x2 -x3", [x1, x2, x3])
        rel = NAryMatrixRelation.from_func_relation(f)

        self.assertEqual(type(rel), NAryMatrixRelation)
        self.assertEqual(len(rel.dimensions), 3)
        self.assertEqual(set(rel.dimensions), {x1, x2, x3})

        for assignmnent in generate_assignment_as_dict([x1, x2, x3]):
            r_val = rel(**assignmnent)
            f_val = f(**assignmnent)
            self.assertEqual(f_val, r_val)

    def test_binary_rel(self):

        x1 = Variable("x1", [1, 2, 3, 4])
        x2 = Variable("x2", [1, 2, 3, 4])
        f = relation_from_str("f", "x1 - x2 ", [x1, x2])
        rel = NAryMatrixRelation.from_func_relation(f)

        self.assertEqual(type(rel), NAryMatrixRelation)
        self.assertEqual(len(rel.dimensions), 2)
        self.assertEqual(set(rel.dimensions), {x1, x2})

        for assignmnent in generate_assignment_as_dict([x1, x2]):
            r_val = rel(**assignmnent)
            f_val = f(**assignmnent)
            self.assertEqual(f_val, r_val)

    def test_binary_hard_rel(self):

        x1 = Variable("x1", [0, 1, 2])
        x2 = Variable("x2", [0, 1, 2])
        f = relation_from_str("f", "10000 if x1 == x2 else 0", [x1, x2])
        rel = NAryMatrixRelation.from_func_relation(f)

        self.assertEqual(type(rel), NAryMatrixRelation)
        self.assertEqual(len(rel.dimensions), 2)
        self.assertEqual(set(rel.dimensions), {x1, x2})

        for assignmnent in generate_assignment_as_dict([x1, x2]):
            r_val = rel(**assignmnent)
            f_val = f(**assignmnent)
            self.assertEqual(f_val, r_val)


class NAryMatrixRelationOtherTests(unittest.TestCase):
    def test_eq(self):

        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation([x1, x2], [[2, 16], [4, 32], [8, 64]])
        u2 = NAryMatrixRelation([x1, x2], [[2, 16], [4, 32], [8, 64]])

        self.assertEqual(u1, u2)

    def test_simple_repr(self):
        x1, x2, u1 = get_2var_rel()

        r = simple_repr(u1)
        print(r)
        self.assertIsNotNone(r)

    def test_from_repr(self):
        x1, x2, u1 = get_2var_rel()

        r = simple_repr(u1)
        print(r)
        u = from_repr(r)
        self.assertEqual(u1, u)

    def test_hash(self):
        x1, x2, u1 = get_2var_rel()

        h = hash(u1)
        self.assertIsNotNone(h)

        u2 = NAryMatrixRelation([x1, x2], [[1, 2], [3, 4], [5, 6]])

        self.assertEqual(h, hash(u2))

        u3 = NAryMatrixRelation([x1, x2], [[1, 5], [3, 4], [5, 6]])

        self.assertNotEqual(h, hash(u3))


class NeutralRelationTest(unittest.TestCase):
    def test_two_vars(self):
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])

        r1 = NeutralRelation([v1, v2])

        self.assertEqual(r1(v1=2, v2=10), 0)
        self.assertEqual(r1(v1=9, v2=42), 0)

    def test_slice(self):
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])

        r1 = NeutralRelation([v1, v2])
        rs1 = r1.slice({"v1": 3})

        self.assertEqual(rs1(v2=10), 0)
        self.assertEqual(rs1(v2=42), 0)
        self.assertEqual(rs1.dimensions, [v2])

    def test_simple_repr(self):
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])

        r1 = NeutralRelation([v1, v2])
        r = simple_repr(r1)

        self.assertEqual(len(r["variables"]), 2)

    def test_from_repr(self):
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])

        r1 = NeutralRelation([v1, v2])
        r = simple_repr(r1)
        r2 = from_repr(r)

        self.assertEqual(r1, r2)

    def test_hash(self):
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])
        r1 = NeutralRelation([v1, v2])

        h = hash(r1)
        self.assertIsNotNone(h)

        r2 = NeutralRelation([v1])
        self.assertNotEqual(h, hash(r2))


class ConditionalRelationsTest(unittest.TestCase):
    def test_dimensions_distinct(self):
        # test when the sets of variables for the condition and the relation
        # are distinct
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", ["R", "G", "B"])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        # With one variables
        @AsNAryFunctionRelation(v2)
        def rel1(v):
            if v == "G":
                return 1
            return 5

        cond_rel = ConditionalRelation(condition, rel1)

        self.assertEqual(len(cond_rel.dimensions), 2)
        self.assertIn(v1, cond_rel.dimensions)
        self.assertIn(v2, cond_rel.dimensions)

        # With Two variables
        v3 = Variable("v3", ["R", "G", "B"])

        @AsNAryFunctionRelation(v2, v3)
        def rel2(v, w):
            if v == w:
                return 10
            elif v == "R":
                return 5
            return 2

        cond_rel2 = ConditionalRelation(condition, rel2)

        self.assertEqual(len(cond_rel2.dimensions), 3)
        self.assertIn(v1, cond_rel2.dimensions)
        self.assertIn(v2, cond_rel2.dimensions)
        self.assertIn(v3, cond_rel2.dimensions)

    def test_dimensions_common(self):
        # test when some variable are used both in the condition and the
        # relation
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", ["R", "G", "B"])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        # With one variables
        @AsNAryFunctionRelation(v1, v2)
        def rel1(v1_, v2_):
            return 5

        cond_rel = ConditionalRelation(condition, rel1)

        self.assertEqual(len(cond_rel.dimensions), 2)
        self.assertIn(v1, cond_rel.dimensions)
        self.assertIn(v2, cond_rel.dimensions)

        # With Two variables in the condition
        @AsNAryFunctionRelation(v1, v2)
        def condition(v1_, v2_):
            return v1_ or v2_

        @AsNAryFunctionRelation(v2)
        def rel2(_):
            return 2

        cond_rel2 = ConditionalRelation(condition, rel2)

        self.assertEqual(len(cond_rel2.dimensions), 2)
        self.assertIn(v1, cond_rel2.dimensions)
        self.assertIn(v2, cond_rel2.dimensions)

    def test_get_val(self):

        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        @AsNAryFunctionRelation(v2, v3)
        def rel1(v2_, v3_):
            return v2_ + v3_

        cond_rel = ConditionalRelation(condition, rel1)

        # get value from a list assignment
        self.assertEqual(5, cond_rel.get_value_for_assignment([True, 2, 3]))
        self.assertEqual(3, cond_rel.get_value_for_assignment([True, 2, 1]))
        self.assertEqual(0, cond_rel.get_value_for_assignment([False, 2, 1]))

        # get value from a dict assignment
        self.assertEqual(
            5, cond_rel.get_value_for_assignment({"v1": True, "v2": 2, "v3": 3})
        )
        self.assertEqual(
            3, cond_rel.get_value_for_assignment({"v1": True, "v2": 2, "v3": 1})
        )
        self.assertEqual(
            0, cond_rel.get_value_for_assignment({"v1": False, "v2": 2, "v3": 3})
        )

        # get value from direct call
        self.assertEqual(5, cond_rel(v1=True, v2=2, v3=3))
        self.assertEqual(3, cond_rel(v1=True, v2=2, v3=1))
        self.assertEqual(0, cond_rel(v1=False, v2=2, v3=3))

    def test_slice_neutral(self):

        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        @AsNAryFunctionRelation(v2, v3)
        def rel1(v2_, v3_):
            return v2_ + v3_

        cond_rel = ConditionalRelation(condition, rel1, return_neutral=True)

        # Slice with true condition
        cond_sliced = cond_rel.slice({v1.name: True})
        self.assertEqual(cond_sliced.dimensions, [v2, v3])
        self.assertEqual(cond_sliced(v2=3, v3=1), 4)
        self.assertEqual(cond_sliced(v2=1, v3=1), 2)

        # Slice with false condition
        cond_sliced = cond_rel.slice({v1.name: False})
        self.assertEqual(cond_sliced.dimensions, [v2, v3])
        self.assertEqual(cond_sliced(v2=3, v3=1), 0)
        self.assertEqual(cond_sliced(v2=1, v3=1), 0)

        # Slice with true condition and v2
        cond_sliced = cond_rel.slice({v1.name: True, v2.name: 1})
        self.assertEqual(cond_sliced.dimensions, [v3])
        self.assertEqual(cond_sliced(v3=3), 4)
        self.assertEqual(cond_sliced(v3=2), 3)

        # Slice with False condition and v2
        cond_sliced = cond_rel.slice({"v1": False, "v2": 2})
        self.assertEqual(cond_sliced.dimensions, [v3])
        self.assertEqual(cond_sliced(v3=3), 0)
        self.assertEqual(cond_sliced(v3=2), 0)

        # Slice while keeping condition and fixing v2
        cond_sliced = cond_rel.slice({"v2": 2})
        self.assertEqual(cond_sliced.dimensions, [v1, v3])
        self.assertEqual(cond_sliced(v1=True, v3=3), 5)
        self.assertEqual(cond_sliced(v1=True, v3=1), 3)
        self.assertEqual(cond_sliced(v1=False, v3=3), 0)
        self.assertEqual(cond_sliced(v1=False, v3=1), 0)

        # Slice while keeping only the condition
        cond_sliced = cond_rel.slice({"v2": 2, "v3": 1})
        self.assertEqual(cond_sliced.dimensions, [v1])
        self.assertEqual(cond_sliced(v1=True), 3)
        self.assertEqual(cond_sliced(v1=False), 0)

    def test_slice_no_neutral(self):
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        @AsNAryFunctionRelation(v2, v3)
        def rel1(v2_, v3_):
            return v2_ + v3_

        cond_rel = ConditionalRelation(condition, rel1, return_neutral=False)

        # Slice with true condition
        cond_sliced = cond_rel.slice({"v1": True})
        self.assertEqual(cond_sliced.dimensions, [v2, v3])
        self.assertEqual(cond_sliced(v2=3, v3=1), 4)
        self.assertEqual(cond_sliced(v2=1, v3=1), 2)

        # Slice with false condition
        cond_sliced = cond_rel.slice({"v1": False})
        self.assertEqual(cond_sliced.dimensions, [])
        self.assertEqual(cond_sliced(), 0)

        # Slice with true condition and v2
        cond_sliced = cond_rel.slice({"v1": True, "v2": 1})
        self.assertEqual(cond_sliced.dimensions, [v3])
        self.assertEqual(cond_sliced(v3=3), 4)
        self.assertEqual(cond_sliced(v3=2), 3)

        # Slice with False condition and v2
        cond_sliced = cond_rel.slice({"v1": False, "v2": 2})
        self.assertEqual(cond_sliced.dimensions, [])
        self.assertEqual(cond_sliced(), 0)

        # Slice while keeping condition and fixing v2
        cond_sliced = cond_rel.slice({"v2": 2})
        self.assertEqual(cond_sliced.dimensions, [v1, v3])
        self.assertEqual(cond_sliced(v1=True, v3=3), 5)
        self.assertEqual(cond_sliced(v1=True, v3=1), 3)
        self.assertEqual(cond_sliced(v1=False, v3=3), 0)
        self.assertEqual(cond_sliced(v1=False, v3=1), 0)

        # Slice while keeping only the condition
        cond_sliced = cond_rel.slice({"v2": 2, "v3": 1})
        self.assertEqual(cond_sliced.dimensions, [v1])
        self.assertEqual(cond_sliced(v1=True), 3)
        self.assertEqual(cond_sliced(v1=False), 0)

    def test_simple_repr(self):
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", ["R", "G", "B"])

        cond_rel = ConditionalRelation(
            relation_from_str("cond", "not v1", [v1]),
            relation_from_str("csq", "v2 * 2", [v2]),
        )

        r = simple_repr(cond_rel)

        self.assertEqual(
            r["condition"], simple_repr(relation_from_str("cond", "not v1", [v1]))
        )
        self.assertEqual(
            r["relation_if_true"], simple_repr(relation_from_str("csq", "v2 * 2", [v2]))
        )

    def test_from_repr(self):
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", ["R", "G", "B"])

        cond_rel = ConditionalRelation(
            relation_from_str("cond", "not v1", [v1]),
            relation_from_str("csq", "v2 * 2", [v2]),
        )

        r = simple_repr(cond_rel)
        cond_rel2 = from_repr(r)

        self.assertEqual(cond_rel, cond_rel2)

    def test_hash(self):
        v1 = Variable("v1", [True, False])
        v2 = Variable("v2", ["R", "G", "B"])

        @AsNAryFunctionRelation(v1)
        def condition(v):
            return v

        # With one variables
        @AsNAryFunctionRelation(v2)
        def rel1(v):
            if v == "G":
                return 1
            return 5

        cond_rel = ConditionalRelation(condition, rel1)

        h = hash(cond_rel)
        self.assertIsNotNone(h)

        cond_rel2 = ConditionalRelation(condition, rel1)
        self.assertEqual(h, hash(cond_rel2))

        @AsNAryFunctionRelation(v2)
        def rel1(v):
            if v == "G":
                return 1
            return 5

        cond_rel3 = ConditionalRelation(condition, rel1)
        self.assertNotEqual(h, hash(cond_rel3))


class CountMatchTests(unittest.TestCase):
    def test_count(self):
        d = list(range(10))
        x0 = Variable("x0", d)
        x1 = Variable("x1", d)
        x2 = Variable("x2", d)

        relation = NAryFunctionRelation(lambda x, y, z: 0, [x0, x1, x2], name="3aryRel")

        count = count_var_match([], relation)
        self.assertEqual(count, 0)

        count = count_var_match([x0.name], relation)
        self.assertEqual(count, 1)

        count = count_var_match([x0.name, x1.name], relation)
        self.assertEqual(count, 2)

        count = count_var_match([x0.name, x1.name, x2.name], relation)
        self.assertEqual(count, 3)


class AssignmentCompatibilityTest(unittest.TestCase):
    def test_disjunct_assignments(self):
        a1 = {"x1": 0, "x3": 1}
        a2 = {"x2": 0, "x4": 1}

        self.assertTrue(is_compatible(a1, a2))

    def test_same_assignments(self):
        a1 = {"x1": 0, "x3": 1}
        a2 = {"x1": 0, "x3": 1}

        self.assertTrue(is_compatible(a1, a2))

    def test_contradicting_assignments(self):
        a1 = {"x1": 0, "x3": 1}
        a2 = {"x1": 1, "x2": 1}

        self.assertTrue(not is_compatible(a1, a2))


class RelationFromExpression(unittest.TestCase):
    def test_relation_from_str_one_var(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2])
        s1 = ExternalVariable("s1", d, 0)
        r = relation_from_str("test_rel", "not not s1", [s1])

        self.assertTrue(r(s1=1))
        self.assertFalse(r(s1=0))
        self.assertEqual(r.expression, "not not s1")

    def test_relation_from_str_2_var(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2])
        s1 = ExternalVariable("s1", d, 0)
        s2 = ExternalVariable("s2", d, 0)
        r = relation_from_str("test_rel", "s1 * s2", [s1, s2])

        self.assertEqual(r(s1=1, s2=3), 3)
        self.assertEqual(r(s2=3, s1=4), 12)
        self.assertIn(s1, r.dimensions)
        self.assertIn(s2, r.dimensions)
        self.assertEqual(2, len(r.dimensions))

        self.assertEqual(r.expression, "s1 * s2")

    def test_relation_from_str_2_boolean_var(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1])
        s1 = ExternalVariable("s1", d, 0)
        s2 = ExternalVariable("s2", d, 0)
        r = relation_from_str("test_rel", "not s1 and s2", [s1, s2])

        self.assertEqual(r(s1=1, s2=0), False)
        self.assertEqual(r(s1=0, s2=1), True)
        self.assertIn(s1, r.dimensions)
        self.assertIn(s2, r.dimensions)
        self.assertEqual(2, len(r.dimensions))

    def test_relation_from_str_multiline(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2])
        s1 = ExternalVariable("s1", d, 0)
        s2 = ExternalVariable("s2", d, 0)
        expr = """
b = s1 / 2
return b * s2        
        """
        r = relation_from_str("test_rel", expr, [s1, s2])

        self.assertEqual(r(s1=1, s2=3), 1.5)
        self.assertEqual(r(s2=3, s1=4), 6)
        self.assertIn(s1, r.dimensions)
        self.assertIn(s2, r.dimensions)
        self.assertEqual(2, len(r.dimensions))

        self.assertEqual(r.expression, expr)


class FindDependentRelations(unittest.TestCase):
    def test_no_relation(self):
        d = VariableDomain("d", "d", [1, 2, 3])
        v1 = Variable("v1", d)

        dependencies = find_dependent_relations(v1, [])
        self.assertEqual(len(dependencies), 0)

    def test_single_relation(self):
        d = VariableDomain("d", "d", [1, 2, 3])
        v1 = Variable("v1", d)

        @AsNAryFunctionRelation(v1)
        def r1(x):
            return x

        dependencies = find_dependent_relations(v1, [r1])
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0], r1)

    def test_independent_relation(self):
        d = VariableDomain("d", "d", [1, 2, 3])
        v1 = Variable("v1", d)
        v2 = Variable("v2", d)

        @AsNAryFunctionRelation(v1)
        def r1(x):
            return x

        dependencies = find_dependent_relations(v2, [r1])
        self.assertEqual(len(dependencies), 0)

    def test_several_relations(self):
        d = VariableDomain("d", "d", [1, 2, 3])
        v1 = Variable("v1", d)
        v2 = Variable("v2", d)
        v3 = Variable("v3", d)

        @AsNAryFunctionRelation(v1, v2)
        def r1(x):
            return x

        @AsNAryFunctionRelation(v1, v3)
        def r2(x):
            return x

        @AsNAryFunctionRelation(v2, v3, v1)
        def r3(x):
            return x

        dependencies_v1 = find_dependent_relations(v1, [r1, r2, r3])
        self.assertEqual(len(dependencies_v1), 3)

        dependencies_v2 = find_dependent_relations(v2, [r1, r2, r3])
        self.assertEqual(len(dependencies_v2), 2)

        dependencies_v3 = find_dependent_relations(v3, [r1, r2, r3])
        self.assertEqual(len(dependencies_v3), 2)

    def test_conditional_relation(self):
        d = VariableDomain("d", "d", [0, 1, 2, 3])

        v1 = Variable("v1", d)
        v2 = Variable("v2", d)
        e1 = ExternalVariable("e1", d, value=0)

        condition = NAryFunctionRelation(lambda x: x, [e1], name="r1_cond")
        rel_if_true = NAryFunctionRelation(lambda x, y: x + y, [v1, v2], name="r1")
        r1 = ConditionalRelation(condition, rel_if_true)

        # even if the condition is not currently active, the dimension of the
        # relation should list all variables
        dependencies = find_dependent_relations(v1, [r1])
        self.assertEqual(len(dependencies), 1)

    def test_conditional_relation_with_initial_values(self):
        d = VariableDomain("d", "d", [0, 1, 2, 3])

        v1 = Variable("v1", d)
        v2 = Variable("v2", d)
        e1 = ExternalVariable("e1", d, value=0)

        condition = NAryFunctionRelation(lambda x: x, [e1], name="r1_cond")
        rel_if_true = NAryFunctionRelation(lambda x, y: x + y, [v1, v2], name="r1")
        r1 = ConditionalRelation(condition, rel_if_true)

        # When the condition is not active with the given external value
        # assignment, the relation should not be listed
        dependencies = find_dependent_relations(v1, [r1], {"e1": 0})
        self.assertEqual(len(dependencies), 0)

        dependencies = find_dependent_relations(v1, [r1], {"e1": 1})
        self.assertEqual(len(dependencies), 1)


class AddVarToRelation(unittest.TestCase):
    def test_add_var_to_zeroary_relation_should_have_arity_one(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2])
        x2 = Variable("x2", d)
        zero_rel = ZeroAryRelation("zeroary", 5)

        def f(x1, r):
            return x1 + r

        new_rel = add_var_to_rel("new_rel", zero_rel, x2, f)

        self.assertEqual(len(new_rel.dimensions), 1)
        self.assertIn(x2, new_rel.dimensions)

    def test_add_var_to_zeroary_relation_should_keep_same_value(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2])
        x2 = Variable("x2", d)
        zero_rel = ZeroAryRelation("zeroary", 5)

        def f(x1, r):
            return x1 + r

        new_rel = add_var_to_rel("new_rel", zero_rel, x2, f)

        # test with kw args and positional args
        self.assertEqual(new_rel(x2=2), 7)
        self.assertEqual(new_rel(2), 7)

    def test_add_var_to_unary_relation_should_have_arity_two(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2, 3, 4, 5])
        x1 = Variable("x1", d)
        x2 = Variable("x2", d)
        unary_rel = UnaryFunctionRelation("unary_rel", x1, lambda x: 2 * x)

        def f(x2, r):
            return x2 + r

        new_rel = add_var_to_rel("new_rel", unary_rel, x2, f)

        self.assertEqual(len(new_rel.dimensions), 2)
        self.assertIn(x2, new_rel.dimensions)
        self.assertIn(x1, new_rel.dimensions)

        # test with kw args and positional args
        self.assertEqual(new_rel(x1=2, x2=3), 7)
        self.assertEqual(new_rel(2, 3), 7)

    def test_add_var_to_n_nary_relation(self):
        d = pydcop.dcop.objects.VariableDomain("d", "d", [0, 1, 2, 3, 4, 5])
        x1 = Variable("x1", d)
        x2 = Variable("x2", d)
        x3 = Variable("x3", d)
        x4 = Variable("x4", d)
        n_ary_rel = NAryFunctionRelation(
            lambda x, y, z: x + 0.1 * y + 0.01 * z, [x1, x2, x3], "n_ary_rel"
        )

        def f(x4, r):
            return 2 * x4 + r

        new_rel = add_var_to_rel("new_rel", n_ary_rel, x4, f)

        self.assertEqual(len(new_rel.dimensions), 4)
        self.assertIn(x1, new_rel.dimensions)
        self.assertIn(x2, new_rel.dimensions)
        self.assertIn(x3, new_rel.dimensions)
        self.assertIn(x4, new_rel.dimensions)

        # test with kw args and positional args
        self.assertEqual(new_rel(x1=1, x2=2, x3=3, x4=4), 9.23)
        # self.assertEqual(new_rel(2, 3), 7)


def test_random_ass_matrix_one_var():
    d = Domain("d", "d", range(3))
    v1 = Variable("v1", d)

    m = random_assignment_matrix([v1], list(range(5)))
    for v in m:
        assert v in range(5)
    print(m)


def test_random_ass_matrix_two_var():
    d = Domain("d", "d", range(4))
    v1 = Variable("v1", d)
    v2 = Variable("v2", d)

    m = random_assignment_matrix([v1, v2], list(range(5)))
    assert m[1][3] in range(5)
    print(m)


def test_assignment_cost_empty():
    assert assignment_cost({}, []) == 0


def test_assignment_cost_one_constraint():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    c1 = constraint_from_str("c1", "v1", [v1])

    assert assignment_cost({"v1": 3}, [c1]) == 3


def test_assignment_cost_one_constraint_two_vars():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    v2 = Variable("v2", domain)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])

    assert assignment_cost({"v1": 3, "v2": 4}, [c1]) == 7


def test_assignment_cost_two_constraints_two_vars():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    v2 = Variable("v2", domain)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])
    c2 = constraint_from_str("c2", "v1*v2", [v1, v2])

    assert assignment_cost({"v1": 2, "v2": 5}, [c1, c2]) == 17


def test_assignment_cost_two_constraints_two_vars_one_extra():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    v2 = Variable("v2", domain)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])
    c2 = constraint_from_str("c2", "v1*v2", [v1, v2])

    assert assignment_cost({"v1": 2}, [c1, c2], v2=5) == 17


def test_assignment_cost_two_constraints_two_costed_vars():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = VariableWithCostFunc("v1", domain, cost_func=lambda x: 0.1 * x)
    v2 = VariableWithCostFunc("v2", domain, cost_func=lambda x: 0.2 * x)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])
    c2 = constraint_from_str("c2", "v1*v2", [v1, v2])

    # We can select if we want to consider variable costs
    assert assignment_cost({"v1": 2, "v2": 5}, [c1, c2]) == 17
    assert (
        assignment_cost({"v1": 2, "v2": 5}, [c1, c2], consider_variable_cost=True)
        == 18.2
    )


def test_assignment_cost_missing_vars():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    v2 = Variable("v2", domain)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])
    c2 = constraint_from_str("c2", "v1*v2", [v1, v2])

    with pytest.raises(KeyError):
        assignment_cost({"v1": 2}, [c1, c2])


def test_assignment_cost_extra_vars():
    domain = VariableDomain("d", "test", list(range(10)))
    v1 = Variable("v1", domain)
    v2 = Variable("v2", domain)
    c1 = constraint_from_str("c1", "v1+v2", [v1, v2])
    c2 = constraint_from_str("c2", "v1*v2", [v1, v2])

    assert assignment_cost({"v1": 2, "v2": 5, "v3": 4}, [c1, c2]) == 17


@pytest.mark.skip
def test_bench_compute_cost(benchmark):
    x1 = Variable("x1", list(range(5)))
    x2 = Variable("x2", list(range(5)))
    x3 = Variable("x3", list(range(5)))
    x4 = Variable("x4", list(range(5)))
    all_vars = [x1, x2, x3, x4]

    c1 = constraint_from_str("c1", "x1 - x2 - 3", all_vars)
    c2 = constraint_from_str("c2", "x1 - x2 -4", all_vars)
    c3 = constraint_from_str("c3", "x4 + x2  -5", all_vars)
    c4 = constraint_from_str("c4", "x3 - x2 -7", all_vars)

    def to_bench():
        assignment_cost(
            {"x1": 3, "x2": 4, "x3": 1, "x4": 2},
            [c1, c2, c3, c4],
            consider_variable_cost=True,
        )

    benchmark(to_bench)


def test_assignment_cost_same_as_becnh():
    x1 = Variable("x1", list(range(5)))
    x2 = Variable("x2", list(range(5)))
    x3 = Variable("x3", list(range(5)))
    x4 = Variable("x4", list(range(5)))
    all_vars = [x1, x2, x3, x4]

    c1 = constraint_from_str("c1", "x1 - x2 - 3", all_vars)
    c2 = constraint_from_str("c2", "x1 - x2 -4", all_vars)
    c3 = constraint_from_str("c3", "x4 + x2  -5", all_vars)
    c4 = constraint_from_str("c4", "x3 - x2 -7", all_vars)

    cost = assignment_cost({"x1": 3, "x2": 4, "x3": 1, "x4": 2}, [c1, c2, c3, c4])

    assert cost == -18


def test_relation_from_str_with_map():
    x1 = Variable("x1", list(range(5)))
    x2 = Variable("x2", list(range(5)))
    all_vars = [x1, x2]

    c1 = constraint_from_str(
        "c1",
        "{('R', 'B'): 0, "
        " ('R', 'R'): 5, "
        " ('B', 'B'): 3, "
        " ('B', 'R'): 1 "
        "}[(x1, x2)]",
        all_vars,
    )

    cost = assignment_cost({"x1": "R", "x2": "B"}, [c1])
    assert cost == 0
    assert (c1(x1="B", x2="B")) == 3


class JoinRelationsTestCase:
    def test_arity_bothsamevar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1])
        u2 = NAryMatrixRelation([x1])

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 1

    def test_arity_2diffvar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1])

        x2 = Variable("x2", ["1", "2"])
        u2 = NAryMatrixRelation([x2])

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 2

    def test_arity_3diffvar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation([x1, x2])

        x3 = Variable("x3", ["z", "y"])
        u2 = NAryMatrixRelation([x2, x3])

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 3

    def test_join_bothsamevar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))
        u2 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))

        # x1 = Variable('x1', ['a', 'b', 'c'])
        # u1 = dpop.NAryRelation([x1], np.array([1, 2, 3], np.int8))

        assert u1.get_value_for_assignment(["b"]) == 2

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 1
        assert u_j.get_value_for_assignment(["a"]) == 2
        assert u_j.get_value_for_assignment(["b"]) == 4
        assert u_j.get_value_for_assignment(["c"]) == 6

    def test_join_2diffvar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        x2 = Variable("x2", ["1", "2"])
        u2 = NAryMatrixRelation([x2], np.array([1, 3], np.int8))

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 2

        assert u_j.get_value_for_assignment(["a", "1"]) == 3
        assert u_j.get_value_for_assignment(["c", "2"]) == 11
        assert u_j.get_value_for_assignment(["b", "1"]) == 5

    def test_join_3diffvar(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation(
            [x1, x2], np.array([[2, 16], [4, 32], [8, 64]], np.int8)
        )

        x3 = Variable("x3", ["z", "y"])
        u2 = NAryMatrixRelation([x2, x3], np.array([[1, 5], [3, 7]], np.int8))

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 3
        assert u_j.dimensions, [x1, x2, x3]

        assert u_j.get_value_for_assignment(["a", "1", "z"]) == 3
        assert u_j.get_value_for_assignment(["b", "2", "y"]) == 39

    def test_join_with_no_var_rel(self):
        # join a relation with a relation with no dimension

        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation(
            [x1, x2], np.array([[2, 16], [4, 32], [8, 64]], np.int8)
        )
        u2 = NAryMatrixRelation([])

        u_j = pydcop.dcop.relations.join(u1, u2)

        assert u_j.arity == 2
        assert u_j.dimensions == [x1, x2]

        assert u_j.get_value_for_assignment(["a", "1"]) == 2
        assert u_j.get_value_for_assignment(["b", "2"]) == 32
        assert u_j(x1="a", x2="1") == 2
        assert u_j(x1="b", x2="2") == 32

    def test_join_different_order(self):
        # Test joining 2 relations that do not declare their variable in the
        # same order

        x1 = Variable("x1", [0, 1, 2])
        x2 = Variable("x2", [0, 1, 2])

        @AsNAryFunctionRelation(x1, x2)
        def u1(x, y):
            return x + y

        @AsNAryFunctionRelation(x2, x1)
        def u2(x, y):
            return x - y

        j = pydcop.dcop.relations.join(u1, u2)

        assert j(1, 1) == 2
        assert j(1, 2) == 4
        assert j(x1=1, x2=1) == 2
        assert j(x1=1, x2=2) == 4


class ProjectionTestCase(unittest.TestCase):
    def test_projection_oneVarRel(self):

        # u1 is a relation with a single variable :
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        # take the projection of u1 along x1
        p = pydcop.dcop.relations.projection(u1, x1)

        # the dimension must be one less than the dimension of u1
        assert p.arity == 0

        # this means that p is actually a signle value, corresponding to the
        # max of u1
        assert p.get_value_for_assignment() == 8

    def test_projection_min_oneVarRel(self):
        # u1 is a relation with a single variable :
        x1 = Variable("x1", ["a", "b", "c"])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        # take the projection of u1 along x1
        p = pydcop.dcop.relations.projection(u1, x1, mode="min")

        # the dimension must be one less than the dimension of u1
        assert p.arity == 0

        # this means that p is actually a signle value, corresponding to the
        # max of u1
        assert p.get_value_for_assignment() == 2

    def test_projection_twoVarsRel(self):

        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation(
            [x1, x2], np.array([[2, 16], [4, 32], [8, 64]], np.int8)
        )

        # take the projection of u1 along x1
        p = pydcop.dcop.relations.projection(u1, x1)

        # the dimension must be one less than the dimension of u1, it should
        # contain only x2
        assert p.arity == 1
        assert p.dimensions == [x2]

        # the max of u1 when setting x2<-1 is 8
        assert p.get_value_for_assignment(["1"]) == 8

        # the max of u1 when setting x2<-2 is 64
        assert p.get_value_for_assignment(["2"]) == 64

    def test_projection_min_twoVarsRel(self):
        x1 = Variable("x1", ["a", "b", "c"])
        x2 = Variable("x2", ["1", "2"])
        u1 = NAryMatrixRelation(
            [x1, x2], np.array([[2, 16], [4, 32], [8, 64]], np.int8)
        )

        # take the projection of u1 along x1
        p = pydcop.dcop.relations.projection(u1, x1, mode="min")

        # the dimension must be one less than the dimension of u1, it should
        # contain only x2
        assert p.arity == 1
        assert p.dimensions == [x2]

        # the min of u1 when setting x2<-1 is 2
        assert p.get_value_for_assignment(["1"]) == 2

        # the min of u1 when setting x2<-2 is 16
        assert p.get_value_for_assignment(["2"]) == 16
