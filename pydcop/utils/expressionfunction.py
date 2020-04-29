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

import ast
import importlib.util
import sys

from typing import List, Tuple, Any, Set
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

    def __init__(self, expression: str, source_file=None, **fixed_vars) -> None:
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
        self._expression = expression.lstrip()
        self._fixed_vars = fixed_vars
        self._source_file = source_file

        has_return, self.exp_vars = _analyse_ast(self._expression)

        # Build the function definition code from the expression:
        f_def = f"def f({', '.join([v for v in self.exp_vars])} ):\n"
        if not has_return:
            f_def += f"    return {self._expression}"
        else:
            self._expression = f"\n{self._expression}" \
                if not self._expression.startswith("\n") \
                else self._expression
            f_def += self._expression.replace("\n", "\n    ")
        try:
            f_compiled = compile(f_def, '<string>', 'exec')
        except SyntaxError:
            raise SyntaxError(f"Syntax error in string expression: '{self._expression}'")

        # Make the module that contains the constraint definition available to exec:
        g = dict(globals())
        if source_file is not None:
            # import the module that contains the constraint definition
            spec = importlib.util.spec_from_file_location("source", source_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            g["source"] = module
        # And execute on compiled function definition, to get the function object:
        try:
            local = {}
            exec(f_compiled, g, local)
            self.exp_func = local["f"]
        except SyntaxError:
            raise SyntaxError(f"Syntax error in multi-line string expression {f_def}'")

        for v in fixed_vars:
            if v not in self.exp_vars:
                raise ValueError('Cannot fix variable "{}" which is not '
                                 'present in the expression ""'
                                 .format(v, expression))

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
        return [ v for v in self.exp_vars if v not in self._fixed_vars]

    def partial(self, **kwargs):
        return ExpressionFunction(self.expression, **kwargs)

    def __call__(self, **kwargs):
        # Note that we only accept named arguments !
        l = kwargs.copy()
        l.update(self._fixed_vars)

        received = set(kwargs.keys())
        expected = set(self.variable_names)
        unexpected = received - expected
        missing = expected - received
        if missing:
            raise TypeError(
                "Missing named argument(s) " + str(missing))
        if unexpected:
            raise TypeError(
                "Unexpected argument(s) " + str(unexpected))

        res = self.exp_func(**l)
        return res

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self._expression == other._expression and \
                self._source_file == other._source_file:
            return True
        return False

    def __str__(self):
        return 'ExpressionFunction({})'.format(self._expression)

    def __repr__(self):
        return 'ExpressionFunction({}, {})'.format(self._expression,
                                                   self.exp_vars)

    def __hash__(self):
        return hash((self._expression, tuple(self._fixed_vars.items())))

    def _simple_repr(self):
        r = super()._simple_repr()
        r['fixed_vars'] = simple_repr(self._fixed_vars)
        return r

    @classmethod
    def _from_repr(cls, r):
        fixed_vars = r['fixed_vars']
        del r['fixed_vars']
        args = {k: from_repr(v) for k, v in r.items()
                if k not in ['__qualname__', '__module__']}
        exp_fct = cls(**args, **fixed_vars)
        return exp_fct


class VarCounterVisitor(ast.NodeVisitor):
    """ A simple visitor to count variables in an AST tree."""

    def __init__(self):
        self.loaded = set()
        self.stored = set()
        self.has_return = False
        self.imported = set()

    def visit(self, node) -> Any:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.loaded.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                self.stored.add(node.id)
        elif isinstance(node, ast.Return):
            self.has_return = True
        # We must keep track of importer name in order to avoid considering as variable
        # names:
        elif isinstance(node, ast.Import):
            self.imported.update([ n.name for n in node.names])
        elif isinstance(node, ast.ImportFrom):
            self.imported.update([ n.name for n in node.names])

        self.generic_visit(node)

    def get_vars(self):
        names = (self.loaded - self.stored) -self.imported
        # We want to allow using builtin function like abs, round, etc.
        # We must filter them out from the list of variables.
        # We also filter out any name that starts with 'source', as this is the syntax
        # used in yaml when referring to constraints defined in separate python files.
        builtins = dir(sys.modules["builtins"])
        return { n for n in names if not n.startswith("source") and n not in builtins}


def _analyse_ast(str_code: str) -> Tuple[bool, Set[str]]:
    """
    Analyse the AST built from `str_definition`.

    Parameters
    ----------
    str_code: str
        A string containing a piece of valid python code : statement, expression or
         function definition (but without the `def ....` line).

    Returns
    -------
    has_return: bool
        True is the expression contains at least one return statement.
    variables: Set of str
        A set containing the identifiers of all variables used but not declared
        in `str_code`.

    """
    node = ast.parse(str_code)
    visitor = VarCounterVisitor()
    visitor.visit(node)
    return visitor.has_return, visitor.get_vars()
