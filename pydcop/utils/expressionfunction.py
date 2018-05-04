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


from typing import List
from collections.abc import Callable
from pydcop.utils.simple_repr import SimpleRepr, simple_repr, from_repr


class ExpressionFunction(Callable, SimpleRepr):
    """
    Callable object representing a function from a python string.
    expression.

    Example:
    f = ExpressionFunction('a + b')
    f.variable_names  -> ['a', 'b']
    f(a=1, b=3)       -> 4
    f.expression      -> 'a + b'

    Note: this callable only works with keyword arguments.

    """

    def __init__(self, expression: str, **fixed_vars) -> None:
        """
        Create a callable representing the expression.

        :param expression: a valid python expression (any builtin python
        function can be used, e.g. abs, round, etc.).
        for example "abs(a1 - b)"

        :param fixed_vars: extra keyword parameters will be interpreted as
        fixed parameter for the expression and the produced callable will
        represent a partial evaluation if the expression with these
        parameter already fixed. If the name of these keyword parameter do
        not match any of the variables found in the expression,
        a `ValueError` is raised.
        """
        self._expression = expression
        self._fixed_vars = fixed_vars

        try:
            self._c = compile('_fres=' + str(expression), '<string>', 'exec')
        except SyntaxError:
            raise SyntaxError('Syntax error in string expression ' +
                              str(expression))
        f_vars = set(self._c.co_names)
        f_vars.remove('_fres')
        for v in fixed_vars:
            if v not in f_vars:
                raise ValueError('Cannot fix variable "{}" which is not '
                                 'present in the expression ""'
                                 .format(v, expression))

        # We want to allow using builtin function like abs, round, etc.
        # We must filter them out from the list of variables
        import sys
        builtins = dir(sys.modules["builtins"])
        self._vars = [v for v in f_vars if v not in builtins]

    @property
    def expression(self):
        return self._expression

    @property
    def __name__(self):
        return self._expression

    @property
    def variable_names(self) -> List[str]:
        """
        :return: a set of variable names that must be set when calling f
        """
        return [ v for v in self._vars if v not in self._fixed_vars]

    def partial(self, **kwargs):
        return ExpressionFunction(self.expression, **kwargs)

    def __call__(self, *args, **kwargs):
        l = kwargs.copy()
        l.update(self._fixed_vars)
        exec(self._c, globals(), l)
        return l['_fres']

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self._expression == other._expression:
            return True
        return False

    def __str__(self):
        return 'ExpressionFunction({})'.format(self._expression)

    def __repr__(self):
        return 'ExpressionFunction({}, {})'.format(self._expression,
                                                   self._vars)

    def __hash__(self):
        return hash((self._expression, tuple(self._fixed_vars.items())))

    def _simple_repr(self):
        r = super()._simple_repr()
        r['fixed_vars'] = simple_repr(self._fixed_vars)
        return r

    @classmethod
    def _from_repr(cls, r):
        fixed_vars =  r['fixed_vars']
        del r['fixed_vars']
        args = {k: from_repr(v) for k, v in r.items()
                if k not in ['__qualname__', '__module__']}
        exp_fct =  cls(**args, **fixed_vars)
        return exp_fct
