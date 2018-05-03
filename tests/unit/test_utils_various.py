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

from pydcop.utils.various import func_args
from pydcop.utils.expressionfunction import  ExpressionFunction


class FuncArgsTests(unittest.TestCase):

    def test_one_arg(self):

        def f(a):
            return a*2
        var_list = func_args(f)

        self.assertEqual(var_list, ['a'])

    def test_two_arg(self):

        def f(a, b):
            return a +b
        var_list = func_args(f)

        self.assertEqual(var_list, ['a', 'b'])

    def test_one_partial(self):
        def f(a, b, c):
            return a + b

        var_list = func_args(f)
        self.assertEqual(var_list, ['a', 'b', 'c'])

        from functools import partial
        f2 = partial(f,b=2)

        var_list = func_args(f2)
        self.assertEqual(var_list, ['a', 'c'])

    def test_one_partial_twice(self):
        def f(a, b, c):
            return a + b

        var_list = func_args(f)
        self.assertEqual(var_list, ['a', 'b', 'c'])

        from functools import partial
        f2 = partial(f, b=2)
        f3 = partial(f2, a=2)

        var_list = func_args(f3)
        self.assertEqual(var_list, ['c'])

    def test_lambda(self):
        f = lambda a, b: a + b

        var_list = func_args(f)

        self.assertEqual(var_list, ['a', 'b'])

    def test_expression_function(self):

        f = ExpressionFunction('a + b + v1')
        var_list = func_args(f)

        self.assertIn('a', var_list)
        self.assertIn('b', var_list)
        self.assertIn('v1', var_list)
        self.assertEqual(len(var_list), 3)
