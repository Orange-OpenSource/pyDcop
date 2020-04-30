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
from functools import partial

import pytest

from pydcop.utils.expressionfunction import ExpressionFunction, _analyse_ast
from pydcop.utils.simple_repr import simple_repr, from_repr


class TestExpressionFunction(unittest.TestCase):

    def test_callable(self):

        f = ExpressionFunction('a / b ')
        self.assertEqual(f(a=4, b=2), 2)

    def test_simple_math_expression(self):

        f = ExpressionFunction('a + b ')
        self.assertEqual(f.expression, 'a + b ')

    def test_oneline_python_expression(self):

        f = ExpressionFunction(' "ko" if a+b > 10 else a+b')

        self.assertEqual(f(a=2, b=3), 5)
        self.assertEqual(f(a=4, b=8), "ko")

    def test_complex_oneline_exp(self):

        # This kind of expression is exactly what we use when modelling an
        # hard constraint:
        f = ExpressionFunction('0 if round(0.2*a + 0.5*b + 0.8*c) == M '
                               'else 1000')

        self.assertEqual(f(a=10, b=10, c=10, M=15), 0)
        self.assertEqual(f(a=10, b=10, c=10, M=13), 1000)
        self.assertEqual(f(a=5, b=2, c=3, M=4), 0)

    def test_variable_names(self):

        f = ExpressionFunction('a + b ')
        names = f.variable_names

        self.assertEqual(len(list(names)), 2)
        self.assertIn('a', names)
        self.assertIn('b', names)

    def test_should_work_with_partial(self):

        f = ExpressionFunction('a * (b -c)')

        fp = partial(f, c=2)
        self.assertEqual(f(a=2, b=5, c=2), 6)
        self.assertEqual(f(a=2, b=5, c=2), fp(a=2, b=5))

        fp = partial(f, c=1, a=3)
        self.assertEqual(f(a=3, b=5, c=1), fp(b=5))

    def test_non_numeric_variable(self):

        f = ExpressionFunction("1 if a == 'A' else 2")
        self.assertEqual(f(a='A'), 1)
        self.assertEqual(f(a='B'), 2)

    def test_str_with_function_call(self):
        r = ExpressionFunction('abs(s1 - s2)')

        self.assertEqual(len(list(r.variable_names)), 2)
        self.assertEqual(r(s1=2, s2=3), 1)
        self.assertEqual(r(s1=3, s2=2), 1)

    def test_raise_on_syntax_error(self):
        self.assertRaises(SyntaxError, ExpressionFunction, '(s1 - s2')

    def test_simple_repr(self):
        f = ExpressionFunction('a + b ')
        self.assertEqual(f.expression, 'a + b ')

        r = simple_repr(f)

        self.assertEqual(r['expression'], 'a + b ')

    def test_from_simple_repr(self):
        f = ExpressionFunction('a + b ')
        self.assertEqual(f.expression, 'a + b ')

        r = simple_repr(f)
        f2 = from_repr(r)

        self.assertEqual(f(a=2, b=3), f2(a=2, b=3))
        self.assertEqual(f2(a=2, b=3), 5)
        self.assertEqual(f, f2)

    def test_partial(self):
        f = ExpressionFunction('a + b ')
        fp = f.partial(a=2)
        self.assertEqual(fp(b=3), 5)

        self.assertNotIn('a', fp.variable_names)

    def test_fixed_vars(self):
        f = ExpressionFunction('a + b ', b=3)
        self.assertEqual(f(a=5), 8)

        self.assertNotIn('b', f.variable_names)

    def test_simple_repr_on_partial(self):
        f = ExpressionFunction('a + b')
        fp = f.partial(a=2)

        r = simple_repr(fp)
        print(r)

        self.assertEqual(r['expression'], 'a + b')
        self.assertIn('a', r['fixed_vars'])
        self.assertEqual(r['fixed_vars']['a'], 2)

    def test_from_repr_on_partial(self):
        f = ExpressionFunction('a + b')
        fp = f.partial(a=2)

        r = simple_repr(fp)
        f1 = from_repr(r)
        print(r)

        self.assertEqual(f1(b=3), 5)
        self.assertEqual(f1(b=5), f(a=2, b=5))

    def test_hash(self):
        f = ExpressionFunction('a + b')
        h = hash(f)

        self.assertEqual(h, hash(ExpressionFunction('a + b')))
        self.assertNotEqual(h, hash(ExpressionFunction('a + c')))

    def test_hash_fixed_vars(self):
        f1 = ExpressionFunction('a + b', b=1)
        f2 = ExpressionFunction('a + b', b=2)
        f3 = ExpressionFunction('a + b')

        self.assertNotEqual(hash(f1), hash(f2))
        self.assertNotEqual(hash(f1), hash(f3))


def test_type_error_on_incomplete_assignment():
    f = ExpressionFunction('a / b ')

    with pytest.raises(TypeError):
        f(a=4)


def test_type_error_on_excessive_assignment():
    f = ExpressionFunction('a / b ')

    with pytest.raises(TypeError):
        f(a=4, b=3, c=2)


def test_analyse_ast_simple_expr_no_variable():
    has_return, exp_vars = _analyse_ast("3 ")
    assert not has_return
    assert exp_vars == set()


def test_analyse_ast_simple_expr_one_variable():
    has_return, exp_vars = _analyse_ast("a + 3 ")
    assert not has_return
    assert exp_vars == { "a"}


def test_analyse_ast_simple_expr_two_variable():
    has_return, exp_vars = _analyse_ast("a + b ")
    assert not has_return
    assert exp_vars == { "a", "b"}


def test_analyse_ast_simple_if_expr():
    has_return, exp_vars = _analyse_ast("10 if a == b  else 0")
    assert not has_return
    assert exp_vars == { "a", "b"}


def test_analyse_ast_func_no_variable():
    has_return, exp_vars = _analyse_ast("""
a = 3
return a
""")
    assert has_return
    assert exp_vars == set()


def test_analyse_ast_func_one_variable():
    has_return, exp_vars = _analyse_ast("""
a = 3
return a + b
""")
    assert has_return
    assert exp_vars == {"b"}


def test_analyse_ast_func_two_variable():
    has_return, exp_vars = _analyse_ast("""
c = 10 *a + 5 * b
return c
""")
    assert has_return
    assert exp_vars == {"a", "b"}


def test_multiline_expression_starting_with_newline():
    exp = ExpressionFunction("""
a=3
return a""")

    assert exp() == 3


def test_multiline_expression_no_newline_at_start():
    exp = ExpressionFunction("""a=3
return a + 2""")

    assert exp() == 5

    # As f has no arg, it must raise an error :
    with pytest.raises(TypeError):
        exp(a=4, b=3, c=2)
    with pytest.raises(TypeError):
        exp(a=4)
    with pytest.raises(TypeError):
        exp(4)


def test_multiline_expression_one_var():
    exp = ExpressionFunction("""a=3
return a * b""")

    assert exp(b=2) == 6

    with pytest.raises(TypeError) as exception:
        exp()
    assert "Missing named argument(s)" in str(exception.value)

    with pytest.raises(TypeError) as exception:
        exp(4)
    assert "takes 1 positional argument but 2 were given" in str(exception.value)


def test_multiline_expression_with_import():
    exp = ExpressionFunction("""import math
return math.pi + a""")

    import math
    assert exp(a=2) == math.pi + 2

    with pytest.raises(TypeError) as exception:
        exp(2)


def test_multiline_expression_with_fromimport():
    exp = ExpressionFunction("""from math import pi
return pi + a""")

    import math
    assert exp(a=2) == math.pi + 2

    with pytest.raises(TypeError) as exception:
        exp(2)