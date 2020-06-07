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


import functools
import random
from copy import deepcopy

import numpy as np
from typing import Dict, Iterable, Any, Tuple, Callable, List, Union

from pydcop.dcop.objects import Variable
from pydcop.utils.simple_repr import SimpleRepr
from pydcop.utils.various import func_args
from pydcop.utils.expressionfunction import ExpressionFunction


DEFAULT_TYPE = np.int32


class RelationProtocol(object):
    """
    This class is used to define a protocol that must be implemented by any
    object that represents a Relation. It is meant to be usable with many
    algorithms, like for example dpop and maxsum.

    It is mostly defined for documentation purpose, objects used for are not
    required to inherit from this class as long as they implement the methods
    defined here.

    Relation objects MUST be immutable and impletemnt `__eq__()` and
    `__hash__()`. This means that methods like  `set_value_for_assignment`
    must return a new relation instead of modifying the current one.
    """

    @property
    def name(self) -> str:
        raise NotImplemented("name not implemented")

    @property
    def dimensions(self) -> List[Variable]:
        """
        The Dimensions of a relation is the list of variables it depends on.
        :return: a list of Variables objects this Relation depends on.
        """
        raise NotImplemented("dimensions not implemented")

    @property
    def scope_names(self) -> List[str]:
        """
        The names of the variable in the scope of this constraint.

        Returns
        -------

        """
        return [v.name for v in self.dimensions]

    @property
    def arity(self) -> int:
        """
        The arity of the relation is the number of variables it depends on.
        :return:
        """
        raise NotImplemented("arity not implemented")

    @property
    def shape(self) -> Tuple:
        """
        The shape of a discrete relation is defined as a tuple containing the
        size of the domain of each variable the relation depends on.

        :return a tuple representing the shape of the relation
        """
        raise NotImplemented("shape not implemented")

    def slice(self, partial_assignment: Dict[str, object]) -> "RelationProtocol":
        """
        Slice operation on a relation.

        :param partial_assignment: a dict {var_name: value} containing the
        name and value of all variable to be sliced out of the relation.

        :return: A new relation with a lower (or equal) arity, depending on
        the same variable(s) than the original relation, minus the sliced
        variables.
        """
        raise NotImplemented("slice not implemented")

    def set_value_for_assignment(
        self, assignment: Dict[str, Any], relation_value
    ) -> "RelationProtocol":
        """

        Return a new relation with the same name and the same value for
        every possible assignment except `assignment`, which maps to
        `relation_value`.

        This method is optional: many concrete Relation implements will
        probably not implement it. In that case they should raise an
        `NotImplemented` exception.

        :param assignment: a full assignment for the relation, containing one
        value for each of the variable this relation depends on.
        :param relation_value: the value of the relation for this assignment.

        :return a new Relation object
        """

        raise NotImplemented("set_value_for_assignment not implemented")

    def get_value_for_assignment(self, assignment):
        """
        Get constraint value for an assignment.

        Notes
        -----
        Relying on dimension order (i.e. passing the assignment as a list)
        is fragile and discouraged, use dict or keyword arguments whenever
        possible instead !


        Parameters
        ----------
        assignment: a list of value
            a full assignment for the relation, containing one
            value for each of the variable this relation depends on. It must be
            either a list of values, in the same order as the dimensions of the
            relation, or a dict { var_name: value}

        Returns
        -------
        the value of the relation for this assignment.
        """
        raise NotImplemented("get_value_for_assignment not implemented")

    def __call__(self, *args, **kwargs):
        """
        Get the value of the relation for a given full assignment.

        This method can be used with anonymous or keyword arguments (but not
        at the same time).

        When using anonymous arguments (*args), the argument must be the
        value of the variable of the relation, in the same order as returned
        by the `dimensions` property.

        For example, instead of: ``r1.get_value_for_assignment([2, 3])`` one
        can write: ``r1(2, 3)``

        When using keyword arguments the order does not matter, there must be
        one a keyword=value pair for each or the variable of the relation,
        the keyword is the name of the variable.

        """
        raise NotImplemented("slice not implemented")


Constraint = RelationProtocol


class AbstractBaseRelation(RelationProtocol):
    """
    This class is meant to be used as a base when implementing a Relation.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._variables = []  # type: List[Variable]

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> List[Variable]:
        return self._variables

    @property
    def arity(self) -> int:
        return len(self._variables)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple([len(v.domain) for v in self._variables])

    def __str__(self):
        return "Relation: {}  on {} ".format(self._name, self._variables)


class ZeroAryRelation(AbstractBaseRelation, SimpleRepr):
    """
    A relation with no argument !
    """

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name)
        self._variables = []
        self._value = value

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        if not partial_assignment:
            return self
        else:
            raise ValueError(
                "ZeroAryRelation can only be sliced with an "
                "empty partial assignement"
            )

    def set_value_for_assignment(self, assignment, relation_value) -> "ZeroAryRelation":
        if len(assignment) != 0:
            raise ValueError("ZeroAryRelation only accept empty assignment")

        return ZeroAryRelation(self.name, relation_value)

    def get_value_for_assignment(self, assignment):
        if len(assignment) != 0:
            raise ValueError("ZeroAryRelation only accept empty assignment")
        return self._value

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return self._value
        raise ValueError("ZeroAryRelation only accept empty assignment")

    def __str__(self):
        return "ZeroAryRelation({})".format(self.name)

    def __repr__(self):
        return "ZeroAryRelation({}, {})".format(self.name, self._value)

    def __eq__(self, other):
        if type(other) != ZeroAryRelation:
            return False
        if self.name == other.name and self._value == other._value:
            return True
        return False

    def __hash__(self):
        return hash((self.name, self._value))


class UnaryFunctionRelation(AbstractBaseRelation, SimpleRepr):
    """
    A relation with a single argument, defined from a function.

    Note: when using a lambda for the relation function, __hash__ for two
    `UnaryFunctionRelation` instances will be different (as the hash of two
    lambda are different, even if they contain the same code). If you need
    the hash to be equal, use an `ExpressionFunction` for the relation
    function.

    Parameters
    ----------
    name: str
        name of the relation
    variable: Variable object
        the single variable the `rel_function` depends on.
    rel_function : callable
        single-argument function defining this relation.

    """

    def __init__(
        self,
        name: str,
        variable: Variable,
        rel_function: Union[ExpressionFunction, Callable[[Any], Union[float, int]]],
    ) -> None:
        super().__init__(name)
        self._variable = variable
        self._variables = [variable]
        self._rel_function = rel_function

    @property
    def expression(self) -> str:
        """

        :return: If the function has been build from a python
        string, the string if you need to serialize the relation later.
        """
        if isinstance(self._rel_function, ExpressionFunction):
            return self._rel_function.expression
        raise AttributeError("The function " + str(self._name) + "has no expression !")

    @property
    def function(self) -> Callable[[Any], Union[float, int]]:
        return self._rel_function

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        if not partial_assignment:
            return self
        elif len(partial_assignment) == 1:
            v_name, = partial_assignment.keys()
            if v_name != self._variable.name:
                raise ValueError("Unknown variable when slicing UnaryRelation")

            name = self._name + "_" + v_name
            return ZeroAryRelation(name, self._rel_function(partial_assignment[v_name]))

        raise ValueError("Too many variables when slicing UnaryRelation")

    def get_value_for_assignment(self, assignment) -> Union[float, int]:

        if isinstance(assignment, list):
            if len(assignment) == 1:
                return self._rel_function(assignment[0])
            raise ValueError(
                "Need exactly one argument to get a value from an" " UnaryRelation"
            )
        elif isinstance(assignment, dict):
            return self._rel_function(assignment[self._variable.name])

        raise ValueError("Assignment must be a list or a dict.")

    def set_value_for_assignment(self, assignment, relation_value):
        raise NotImplementedError(
            "Cannot set value on unary factor defined " "with a function"
        )

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self._rel_function(args[0])
        elif len(kwargs) == 1:
            return self._rel_function(kwargs[self._variable.name])
        raise ValueError(
            "Need exactly one argument to get a value from an " "UnaryRelation"
        )

    def __str__(self):
        return "UnaryFunctionRelation({})".format(self._name)

    def __repr__(self):
        return "UnaryFunctionRelation" "({}, {}, {})".format(
            self._name, self._variable, self._rel_function
        )

    def __eq__(self, other):
        if type(other) != UnaryFunctionRelation:
            return False
        if (
            self.name == other.name
            and self._variable == other.dimensions[0]
            and self._rel_function == other.function
        ):
            return True
        return False

    def __hash__(self):
        return hash((self.name, self._variable, self._rel_function))


class UnaryBooleanRelation(AbstractBaseRelation, SimpleRepr):
    """
    A boolean relation with a single argument.
    The value of the relation is the argument, directly interpreted as a
    boolean
    """

    def __init__(self, name: str, var: Variable) -> None:
        """
        :param name: name of the relation
        :param var: Variable object
        """
        super().__init__(name)
        self._var = var
        self._variables = [var]

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        if not partial_assignment:
            return self
        elif len(partial_assignment) == 1:
            v_name, = partial_assignment.keys()
            if v_name != self._var.name:
                raise ValueError(
                    "Unknown variable when slicing " "UnaryBooleanRelation"
                )

            val = True if partial_assignment[v_name] else False
            name = self._name + "_" + v_name
            return ZeroAryRelation(name, val)

        raise ValueError("Invalid slice argument on UnaryBooleanRelation")

    def get_value_for_assignment(self, assignment):

        if isinstance(assignment, list):
            if len(assignment) == 1:
                return True if assignment[0] else False
            raise ValueError(
                "Need exactly one argument to get a value from an" " UnaryRelation"
            )
        elif isinstance(assignment, dict):
            return True if assignment[self._var.name] else False

        raise ValueError("Assignment must be a list or a dict.")

    def set_value_for_assignment(self, assignment, relation_value):
        raise NotImplementedError(
            "Cannot set value on unary factor defined " "with a function"
        )

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return True if args[0] else False
        elif len(kwargs) == 1:
            return True if kwargs[self._var.name] else False
        raise ValueError(
            "Need exactly one argument to get a value from an " "UnaryRelation"
        )

    def __str__(self):
        return "UnaryFunctionRelation({})".format(self._name)

    def __repr__(self):
        return "UnaryFunctionRelation({}, {})".format(self._name, self._var)

    def __eq__(self, other):
        if type(other) != UnaryBooleanRelation:
            return False
        if self.name == other.name and self._var == other.dimensions[0]:
            return True
        return False

    def __hash__(self):
        return hash((self.name, self._var))


class NAryFunctionRelation(AbstractBaseRelation, SimpleRepr):
    """
    A Relation defined from a python function.

    Wrapper used to transform a function into a Relation by implementing the
    Relation protocol.
    It's easier to use it with the @AsNAryFunctionRelation annotation.

    NAryFunctionRelation uses the SimpleRepr mixin, meaning it can be
    serialized (for exemple to transfer it over the network). However,
    this is only true for NAryFunctionRelation that have been created from an
    ExpressionFunction (as the f: Callable parameter).
    If the relation has been created from an arbitrary python, calling
    simple_repr(relation) will raise a SimpleReprException.

    Note: when using a lambda or a python function for the relation function,
    __hash__ for two `NAryFunctionRelation` instances will be different (as
    the hash of two lambdas (or functions) are different, even if they contain
    the same code). If you need the hashes to be equal, use an
    `ExpressionFunction` for the relation function.
    """

    def __init__(
        self,
        f: Callable,
        variables: Iterable[Variable],
        name: str = None,
        f_kwargs=False,
    ) -> None:
        """
        A Relation defined from a python function.

        :param f: a function defining the relation. This can be any python
        function as long as it has only keyword argument. This means you
        cannot use function defined with `f(*args)` but `f(**kwargs)` is fine.
        Use of ExpressinFunction is recommended (and necessary if you need to
        serialize the relation with SimpleRepr).

        :param variables: the list of Variable objects this relation depends
        on. The variable do not need to have the same name than the
        function's arguments but they must be in the same order. If you want
        to use kw args instead of relying on variable order, use the
        f_kwargs argument. In this case, the arguments name
        must map the the variables names. The function can be defined as
        `f(**kwargs)`, even without using f_kwargs, as long as at runtime it
        accepts named arguments with the name of all variables.

        :param name: name of the relation
        :param f_kwargs: If true, we consider that f use the list of
        variables names as keyword arguments. This is useful for function
        that only accept keyword arguments, for the function produced by
        json_serialization.function_from_str.
        """

        try:
            name = name if name is not None else f.__name__
        except AttributeError:
            # function obtained with functools.partial have bo __name__
            name = None
        super().__init__(name)
        self._f = f
        self._variables = list(variables)
        self._f_kwargs = f_kwargs

        # rel var name => function arg name
        self._var_mapping = {}  # type: Dict[str, str]
        if not f_kwargs:
            # build a mapping from the function arguments to the name of the
            # variables of the relation
            var_list = func_args(f)
            if var_list:
                for i, var_name in enumerate(var_list):
                    self._var_mapping[self._variables[i].name] = var_name
            else:
                # If we could not find any arguments with names (which is the
                # case if the function was declared using f( **kwrgs) )
                # default back to use the name of the variables as mapping.
                self._var_mapping = {v.name: v.name for v in variables}

        else:
            self._var_mapping = {v.name: v.name for v in variables}

    @property
    def expression(self):
        """

        :return: If the function has been build from a python
        string, the string if you need to serialize the relation later.
        """
        if hasattr(self._f, "expression"):
            return self._f.expression
        raise AttributeError("The function " + str(self._name) + "has no expression !")

    @property
    def function(self):
        return self._f

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        if not partial_assignment:
            return self
        elif len(partial_assignment) > len(self._variables):
            raise ValueError(
                "Too many many variables when slicing relation "
                "{} : {}".format(self._name, partial_assignment)
            )
        else:
            # Check we're only slicing on existing variables
            _var_names = [v.name for v in self._variables]
            for v in partial_assignment:
                if v not in _var_names:
                    raise ValueError(
                        'Unknown variable "{}" when slicing '
                        "relation {}".format(v, self._name)
                    )

            remaining_vars = [
                v for v in self._variables if v.name not in partial_assignment
            ]
            slicing_dict = {
                self._var_mapping[vn]: partial_assignment[vn]
                for vn in partial_assignment
            }
            if hasattr(self._f, "partial"):
                slice_f = self._f.partial(**slicing_dict)
            else:
                slice_f = functools.partial(self._f, **slicing_dict)

            # Check if there is a source file for the relation defined
            if 'source' in self._f.exp_func.__globals__:
                # Add the source object
                slice_f.exp_func.__globals__['source'] = self._f.exp_func.__globals__['source']

            return NAryFunctionRelation(slice_f, remaining_vars, name=self.name)

    def set_value_for_assignment(self, assignment, relation_value):
        raise NotImplementedError(
            "set_value_for_assignment is not "
            "implemented for function-defined relations"
        )

    def get_value_for_assignment(self, assignment):

        if isinstance(assignment, list):
            args_dict = {}
            for i in range(len(assignment)):
                arg_name = self._var_mapping[self._variables[i].name]
                args_dict[arg_name] = assignment[i]
            return self._f(**args_dict)

        elif isinstance(assignment, dict):
            args_dict = {}
            for var_name in assignment:
                arg_name = self._var_mapping[var_name]
                args_dict[arg_name] = assignment[var_name]
            return self._f(**args_dict)

        else:
            raise ValueError("Assignment must be list or dict")

    def __call__(self, *args, **kwargs):
        if not kwargs:
            if len(args) == 1 and type(args[0]) is dict:
                return self(**args[0])
            return self.get_value_for_assignment(list(args))
        else:
            return self.get_value_for_assignment(kwargs)

    def __repr__(self):
        return "NAryFunctionRelation({}, {})".format(self.name, self._variables)

    def __str__(self):
        return "NAryFunctionRelation({})".format(self._name)

    def __eq__(self, other):
        if type(other) != NAryFunctionRelation:
            return False
        if (
            self.name == other.name
            and other.dimensions == self.dimensions
            and self._f == other.function
        ):
            return True
        return False

    def __hash__(self):
        return hash((self.name, tuple(self._variables), self._f))


class AsNAryFunctionRelation(object):
    """
    The AsNAryFunctionRelation decorator can be used to transform any function
    into an object implementing the Relation protocol.
    This annotation requires an argument giving the list of variable (as
    Variable objects) that the relation depends on. The names of arguments of
    the function do not need to be same as the Variable's names but the order
    must be same.

    Example :
        from algorithms import Variable, AsNAryFunctionRelation
        domain = list(range(10))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x, y):
            return x + y

    Note: as relations created with this annotation use the python function
    as the relation function, the hash() of two of these relations will be
    different even if their function is exactly the same.

    """

    def __init__(self, *args) -> None:
        self.vars = list(args)

    def __call__(self, f):

        return NAryFunctionRelation(f, self.vars)


class NAryMatrixRelation(AbstractBaseRelation, SimpleRepr):
    """
    This class represent a n-ary relation other a set of variable xi.

    It is a multi dimensional matrix, with one dimension for each variable in
    the separator of variable the variable sending this messsage.

    Notes
    -----

    Note: when defining the relation with a matrix, the dimension of the
    matrix must be in the same order as the variable. For a 2-ary
    relation, the first variables maps to the column of the matrix,
    and the second variable to the rows.

    The matrix is and array of array (of array... depending on the arity
    of the relation). The inner most arrays maps to the values of the
    last variables (and thus have the same size than the domain of the
    last variables).

    For example :

        x1 = dpop.Variable('x1', ['a', 'b', 'c'])
        x2 = dpop.Variable('x2', ['1', '2'])
        u1 = dpop.NAryRelation([x1, x2], [[2, 16],
                                          [4, 32],
                                          [8, 64]])

    Parameters
    ----------
    variables: a list or finite iterable of variables
        the variables this relation depends on (its
        dimensions)
    matrix: an optional array or np.array matrix
        an optional array or np.array matrix those dimension
        maps the domains of the variables of the relation. If the matrix is
        not given, the relation returns 0.0 for all assignments of the
        variables. Notice that we use float by default when initializing an
        zeroed relation
    name: str
        an optional name for the relation.

    """

    def __init__(
        self, variables: Iterable[Variable], matrix=None, name: str = None
    ) -> None:
        super().__init__(name)
        self._variables = list(variables)
        shape = tuple([len(v.domain) for v in variables])
        if matrix is None:
            # create a zero-filled matrix
            self._m = np.zeros(shape=shape, dtype=np.float64)

        else:
            if not isinstance(matrix, np.array.__class__):
                matrix = np.array(matrix)
            if shape != matrix.shape:
                raise AttributeError(
                    "Invalid dimension when building util " "from matrix"
                )
            self._m = matrix

    def slice(
        self, partial_assignment: Dict[str, object], ignore_extra_vars=False
    ) -> "NAryMatrixRelation":
        if not partial_assignment:
            return self
        sliced_vars, sliced_values = zip(*partial_assignment.items())
        slice_vars, s = self._slice_matrix(
            sliced_vars, sliced_values, ignore_extra_vars=ignore_extra_vars
        )
        u = NAryMatrixRelation(slice_vars, self._m[s], self.name)
        return u

    def _slice_matrix(self, sliced_vars, sliced_values, ignore_extra_vars=False):

        s_vars = list(sliced_vars)
        s_values = list(sliced_values)

        var_names = [v.name for v in self._variables]
        for i, v in enumerate(sliced_vars):
            if v not in var_names:
                if not ignore_extra_vars:
                    raise AttributeError(
                        "{} is not in the dimensions of util : {}".format(
                            v, self._variables
                        )
                    )
                else:
                    del s_vars[i]
                    del s_values[i]

        slices = []
        slice_vars = []
        for v in self._variables:
            if v.name in s_vars:
                slice_index = s_vars.index(v.name)
                val = s_values[slice_index]
                val_index = v.domain.index(val)
                slices.append(val_index)
            else:
                slices.append(slice(None))
                slice_vars.append(v)

        return slice_vars, tuple(slices)

    def get_value_for_assignment(self, var_values=None):
        """
        Returns the value of the relation for an assignment.

        :param var_values: either a list or a dict.
        * If var_values is a list, it must be  an array of values
        representing a full assignment of the variables of the relation,
        in the same order as the variables in the dimension.
        * If it is a dict, it must be a var_name => var_value mapping

        :return: the value of the relation.
        """

        if var_values is None:
            if self._m.shape == ():
                return self._m
            else:
                raise KeyError(
                    "Needs an assignement when requesting value "
                    "in a n-ari relation, n!=0"
                )
        if isinstance(var_values, list):
            assignt = {self._variables[i].name: val for i, val in enumerate(var_values)}
            u = self.slice(assignt)
            return u._m.item()

        elif isinstance(var_values, dict):
            u = self.slice(var_values)
            return u._m.item()

        else:
            raise ValueError("Assignment must be dict or array")

    def __call__(self, *args, **kwargs):
        """
        Shortcut method for get_value_for_assignment.
        Instead of using `relation.get_value_for_assignment([val1, val2,
        ...])` you can use the relation as a callable and directly use
        `relation(val1, val2, ...)`

        :param args: values representing a full assignment of
        the variables of the relation, in the same order as the variables in
        the dimension.
        :return: the value of the relation.
        """

        if not kwargs:
            return self.get_value_for_assignment(list(args))
        else:
            return self.get_value_for_assignment(kwargs)

    def set_value_for_assignment(self, var_values, rel_value) -> "NAryMatrixRelation":
        """
        Set the value of the relation for an assignment.

        WARNING: this returns a new relation with the value set for this assignment and
        DOES NOT modify the current relation !!

        :param var_values: either a list or a dict.
        * If var_values is a list, it must be  an array of values
        representing a full assignment of the variables of the relation,
        in the same order as the variables in the dimension.
        * If it is a dict, it must be a var_name => var_value mapping

        :param rel_value: the value of the relation.
        """
        if isinstance(var_values, list):
            _, s = self._slice_matrix([v.name for v in self._variables], var_values)
            matrix = np.copy(self._m)
            matrix[s] = rel_value
            return NAryMatrixRelation(self._variables, matrix, name=self.name)

        elif isinstance(var_values, dict):
            values = []
            for v in self._variables:
                values.append(var_values[v.name])
            _, s = self._slice_matrix([v.name for v in self._variables], values)
            matrix = np.copy(self._m)
            matrix.itemset(s, rel_value)
            return NAryMatrixRelation(self._variables, matrix, name=self.name)
        raise ValueError("Could not set value, must be list or dict")

    @staticmethod
    def from_func_relation(rel: RelationProtocol) -> "NAryMatrixRelation":
        variables = rel.dimensions
        cost_matrix = NAryMatrixRelation(variables)
        # We also compute the min and max value of the constraint as it is to
        # be needed in gdba
        mini = None
        maxi = None

        for asgt in generate_assignment_as_dict(variables):
            value = rel(asgt)
            cost_matrix = cost_matrix.set_value_for_assignment(asgt, value)

        return cost_matrix

    def __str__(self):
        if self._name:
            return f"NAryMatrixRelation({self._name}, {[v.name for v in self._variables]}, {self._m})"
        else:
            return f"NAryMatrixRelation({[v.name for v in self._variables]}, {self._m})"

    def __repr__(self):
        return f"NAryMatrixRelation({self.name}, {[v.name for v in self._variables]}, {self._m})"

    def __eq__(self, other):
        if type(other) != NAryMatrixRelation:
            return False
        if (
            self.name == other.name
            and self.dimensions == other.dimensions
            and np.all(self._m == other._m)
        ):
            return True
        return False

    def __hash__(self):
        # Hack : hashing str(self._m) is not perfect, as it does not take
        # into account the full numpy.ndarray, but it should be enough and
        # is much faster. hash collision are not dramatic in our case.
        return hash((self.name, tuple(self._variables), str(self._m)))

    def _simple_repr(self):
        self._matrix = self._m.tolist()
        r = super()._simple_repr()
        self._matrix = None
        return r


class NeutralRelation(AbstractBaseRelation, SimpleRepr):
    """
    A neutral relation is a relation that always return zero for any value of
    the input variables.
    """

    def __init__(self, variables: Iterable[Variable], name: str = None) -> None:
        super().__init__(name)
        self._variables = list(variables)

    def __str__(self):
        return "NeutralRelation({})".format(self._name)

    def __repr__(self):
        return "NeutralRelation({}, {}".format(self._name, self._variables)

    def __eq__(self, other):
        if type(other) != NeutralRelation:
            return False
        if self.name == other.name and self.dimensions == other.dimensions:
            return True
        return False

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        remaining_vars = [
            v for v in self._variables if v.name not in partial_assignment
        ]
        return NeutralRelation(remaining_vars)

    def get_value_for_assignment(self, assignment):
        return 0

    def __call__(self, *args, **kwargs):
        return 0

    def __hash__(self):
        return hash((self._name, tuple(self._variables)))


class ConditionalRelation(RelationProtocol, SimpleRepr):
    """
    A ConditionalRelation a relation object that implements the Relation
    protocol and is composed of a condition relation and a consequence
    relation.

    the dimension of the Conditional relation is the union of the variables
    used in the two relation it is composed of.

    The condition is evaluated when slicing the ConditionalRelation on the
    variables used in the condition. If the given partial assignments returns
    False when applied to the condition, the slice returns:
    * either a neutral relation on the remaining variable if return_neutral
    is True
    * or a ZeroAryRelation if return_neutral is false


    It can be used for rules for examples.

    """

    def __init__(
        self,
        condition: RelationProtocol,
        relation_if_true: RelationProtocol,
        name: str = None,
        return_neutral: bool = False,
    ) -> None:
        """
        Create a new ConditionalRelation with the given condition and
        consequence.

        :param condition: a relation whose result will be
        interpreted as a boolean.
        :param relation_if_true: relation to be used when the condition is true
        :param name: name of the relation, it None the name of the consequence
        relation will be used.

        :param return_neutral: if true, slicing the relation with a partial
        assignments that maps to a False condition will return a
        NeutralRelation on the remaining variable, otherwise it will return a
        ZeroAryRelation
        """

        self._condition = condition
        self._relation_if_true = relation_if_true
        self._name = name if name is not None else relation_if_true.name
        self._return_neutral = return_neutral

    @property
    def condition(self) -> RelationProtocol:
        return self._condition

    @property
    def consequence(self) -> RelationProtocol:
        return self._relation_if_true

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> List[Variable]:
        dims = list(self._condition.dimensions)
        for v in self._relation_if_true.dimensions:
            if v not in dims:
                dims.append(v)
        dims.sort(key=lambda vo: vo.name)
        return dims

    @property
    def arity(self) -> int:
        return len(self.dimensions)

    @property
    def shape(self) -> Tuple:
        return tuple([len(v.domain) for v in self.dimensions])

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:

        cond_var_names = [v.name for v in self._condition.dimensions]
        true_names = [v.name for v in self._relation_if_true.dimensions]

        cond_args = {
            v_name: v_val
            for v_name, v_val in partial_assignment.items()
            if v_name in cond_var_names
        }

        if len(cond_args) == len(self._condition.dimensions):
            # We have all arguments to evaluate the condition, we can take it
            # out when slicing.
            if self._condition(**cond_args):
                if len(partial_assignment) > len(cond_args):
                    # We have some extra variables to slice the consequence on.
                    slice_dict = {
                        k: v for k, v in partial_assignment.items() if k in true_names
                    }
                    return self._relation_if_true.slice(slice_dict)
                else:
                    return self._relation_if_true
            else:
                if self._return_neutral:
                    remaining_vars = [
                        v
                        for v in self._relation_if_true.dimensions
                        if v.name not in partial_assignment
                    ]
                    return NeutralRelation(remaining_vars)
                else:
                    return ZeroAryRelation(self.name + "_zeroed", 0)
        else:
            if cond_args:
                sliced_cond = self._condition.slice(cond_args)
            else:
                sliced_cond = self._condition

            slice_dict = {
                k: v for k, v in partial_assignment.items() if k in true_names
            }
            if slice_dict:
                sliced_rel = self._relation_if_true.slice(slice_dict)
            else:
                sliced_rel = self._relation_if_true

            return ConditionalRelation(sliced_cond, sliced_rel)

    def get_value_for_assignment(self, assignment):

        if isinstance(assignment, list):

            cond_args = {
                v.name: v_val
                for v, v_val in zip(self.dimensions, assignment)
                if v in self._condition.dimensions
            }
            if self._condition(**cond_args):
                rel_args = {
                    v.name: v_val
                    for v, v_val in zip(self.dimensions, assignment)
                    if v in self._relation_if_true.dimensions
                }
                return self._relation_if_true(**rel_args)
            else:
                return 0

        elif isinstance(assignment, dict):
            cond_args = {v.name: assignment[v.name] for v in self._condition.dimensions}
            if self._condition(**cond_args):
                rel_args = {
                    v.name: assignment[v.name]
                    for v in self._relation_if_true.dimensions
                }
                return self._relation_if_true(**rel_args)
            else:
                return 0
        else:
            raise ValueError("Assignment must be list or dict")

    def __call__(self, *args, **kwargs):
        if not kwargs:
            if len(args) == 1 and type(args[0]) is dict:
                return self(**args[0])
            return self.get_value_for_assignment(list(args))
        else:
            return self.get_value_for_assignment(kwargs)

    def __str__(self):
        return "ConditionalRelation({})".format(self.name)

    def __repr__(self):
        return "ConditionalRelation({} - {} on {} ".format(
            self._condition, self._relation_if_true, self.dimensions
        )

    def __eq__(self, other):
        if type(other) != ConditionalRelation:
            return False
        if (
            self.name == self.name
            and self.dimensions == other.dimensions
            and self.condition == other.condition
            and self.consequence == other.consequence
        ):
            return True
        return False

    def __hash__(self):
        return hash((self.name, self.consequence, self.condition, self._return_neutral))


def count_var_match(var_names, relation):
    """
    Count the number of common variables between agt_vars and the dimensions
    or relation.

    :param var_names: a list of variable names
    :param relation: a relation object
    :return: the number of common variable
    """
    match = 0
    for v in relation.dimensions:
        if v.name in var_names:
            match += 1
    return match


def assignment_matrix(variables: List[Variable], default_value=None):
    """
    Build a matrix for all possible assignment for the variables.

    The matrix is and array of array (of array... depending on the arity
    of the relation). The inner most arrays maps to the values of the
    last variables (and thus have the same size than the domain of the
    last variables).


    Parameters
    ----------
    variables: list of Variables
        the variables used in the assignment.

    default_value
        the default value used in the matrix for all assignment.

    Returns
    -------
    an array of arrays (..of arrays...)

    Notes
    -----

    This can be used to build the matrix used when creating a
    NAryMatrixRelation.
    """
    current = default_value

    for v in reversed(variables):
        tmp = []
        for _ in range(len(v.domain)):
            tmp.append(deepcopy(current))
        current = tmp
    return current


def random_assignment_matrix(variables: List[Variable], values: List, matrix=None):
    """
    Generate a matrix that defines a value for each possible assignment.

    Parameters
    ----------
    variables
    values

    Returns
    -------

    """
    if matrix is None:
        matrix = assignment_matrix(variables)

    if len(variables) == 1:
        for i, _ in enumerate(variables[0].domain):
            matrix[i] = random.choice(values)
    else:
        for i, _ in enumerate(variables[0].domain):
            matrix[i] = random_assignment_matrix(variables[1:], values, matrix[i])

    return matrix


def find_dependent_relations(
    variable: Variable,
    constraints: Iterable[Constraint],
    ext_var_assignment: Dict[str, Any] = None,
) -> Iterable[Constraint]:
    """Find constraints that depends on a given variable.

    Find in `constraints` the constraints that have this variable in their
    scope.     If ext_var_ext_var_assignment is given, we consider relations
    after slicing these variables out. This is useful for relation whose
    dimensions change depending on the value of external variables (like
    ConditionalRelation, for example).

    Parameters
    ----------
    variable: a Variable object
        the variable to look for
    constraints: iterable of Constraints
        set of constraints to look into
    ext_var_assignment: a dict { var_name: value}
        assignment for external variable in these constraints

    Returns
    -------
    list of Constraints objects
    """
    dependent_relations = []
    for r in constraints:
        if variable in r.dimensions:
            if ext_var_assignment:
                s = r.slice(filter_assignment_dict(ext_var_assignment, r.dimensions))
                if len(s.dimensions) > 0:
                    dependent_relations.append(r)
            else:
                dependent_relations.append(r)
    return dependent_relations


def is_compatible(assignment1: Dict[str, Any], assignment2: Dict[str, Any]):
    """
    Check if two (potentially partial) assignments are compatible.
    Compatible means that there is no disagreement on variable assignment.

    :param assignment1: a dict var ->val
    :param assignment2: a dict var ->val
    :return: True is the assignment are compatible
    """
    inter = set(assignment1.keys()) & set(assignment2.keys())
    if len(inter) == 0:
        return True
    for k in inter:
        if assignment1[k] != assignment2[k]:
            return False
    return True


def constraint_from_str(name: str, expression: str, all_variables: Iterable[Variable]):
    """
    Generate a relation object from a string expression and a list of
    available variable objects.

    Example:
    r = relation_from_str('test_rel', 's1 * s2', [s1, s2, s3, l1])

    :param name: name of the relation
    :param expression: python string expression representing the function of
    the relation
    :param all_variables: list of all available variable objects the relation
    could depend on. The exact scope of the relation depends on the content of
    the expression, but any variable used in the expression must be in this
    list.

    :return: a relation object whose function implements the expression
    """
    f_exp = ExpressionFunction(expression)
    relation_variables = []
    for v in f_exp.variable_names:
        found = False
        for s in all_variables:
            if s.name == v:
                relation_variables.append(s)
                found = True
        if not found:
            raise Exception(
                "Missing variable {} for string-based function "
                '"{}"'.format(v, expression)
            )

    return NAryFunctionRelation(f_exp, relation_variables, name, f_kwargs=True)


# We keep relation_from_str as an alias for now, but constraint_from_str
# should be used.
relation_from_str = constraint_from_str

def constraint_from_external_definition(name: str,
        source_file: str, expression: str, all_variables: Iterable[Variable]):

    f_exp = ExpressionFunction(expression, source_file)
    relation_variables = []
    for v in f_exp.variable_names:
        found = False
        for s in all_variables:
            if s.name == v:
                relation_variables.append(s)
                found = True
        if not found:
            raise Exception(
                "Missing variable {} for string-based function "
                '"{}"'.format(v, expression)
            )

    return NAryFunctionRelation(f_exp, relation_variables, name, f_kwargs=True)


def add_var_to_rel(
    name: str, original_relation: RelationProtocol, variable: Variable, f
):
    """
    Create a new relation by adding a variable to the domain of an existing
    relation.

    If original_relation has a arity of n, the produced relation will have an
    arity of n+1.


    :param name: name of the new relation
    :param original_relation: relation the new relation is based on
    :param variable: Variable object that will be included to the scope of
    the new relation.
    :param f:

    :return: a new relation based on original_relation with the extra
    variable in its scope.
    """

    variables = list(original_relation.dimensions) + [variable]

    def new_rel_f(**kwargs):

        args_for_original = {k: v for k, v in kwargs.items() if k != variable.name}
        original_value = original_relation(**args_for_original)

        return f(kwargs[variable.name], original_value)

    return NAryFunctionRelation(new_rel_f, variables, name=name, f_kwargs=True)


def find_optimum(constraint: Constraint, mode: str) -> float:
    """
    Compute the optimum of the relation given the mode.

    Warning: this method enumerate all possible assignments for this relation and will
    be slow, only use it with low arity relation and small domains!

    Parameters
    ----------
    constraint: Constraint
        a constraint object
    mode: str
        'min' or 'max'

    Returns
    -------
    float:
        The best value of the relation, depending on the requested mode,
        as a float.

    """
    if mode != "min" and mode != "max":
        raise ValueError("mode must be 'min' or 'max', not " + str(mode))
    variables = [v for v in constraint.dimensions]
    optimum = None
    for asgt in generate_assignment_as_dict(variables):
        rel_val = constraint(**filter_assignment_dict(asgt, constraint.dimensions))
        if optimum is None:
            optimum = rel_val
        elif mode == "max" and rel_val > optimum:
            optimum = rel_val
        elif mode == "min" and rel_val < optimum:
            optimum = rel_val
    return optimum


def get_data_type_max(data_type):
    # see http://docs.scipy.org/doc/numpy/user/basics.types.html

    if data_type == np.int8:
        return 127
    elif data_type == np.int16:
        return 32767
    elif data_type == np.int32:
        return 2147483647


def get_data_type_min(data_type):

    if data_type == np.int8:
        return -128
    elif data_type == np.int16:
        return -32768
    elif data_type == np.int32:
        return -2147483648


def generate_assignment(variables: List[Variable]):
    """
    Returns a generator iterating on all possible assignments for the set of
    variables vars.

    An assignment is represented as a list of values, in the same order as
    the list of variables.

    Parameters
    ----------

    variables: a list of variable objects.

    Returns
    -------
    a generator iterating on all possible assignments for the set of
    variables vars
    """

    if len(variables) == 0:
        yield []
    else:
        for d in variables[-1].domain:
            for ass in generate_assignment(variables[:-1]):
                ass.append(d)
                yield ass


def generate_assignment_as_dict(variables: List[Variable]):
    """
    Returns a generator iterating on all possible assignments for the set of
    variables vars.

    An assignment is represented as a dict {var_name => var_value}.

    Parameters
    ----------
    variables: a list of variable objects.

    Returns
    -------
    a generator iterating on all possible assignments for the set of
    variables vars
    """

    if len(variables) == 0:
        yield {}
    else:
        current_var = variables[-1]
        for d in current_var.domain:
            for ass in generate_assignment_as_dict(variables[:-1]):
                ass[current_var.name] = d
                yield ass


def assignment_cost(
    assignment: Dict[str, Any],
    constraints: Iterable["Constraint"],
    consider_variable_cost=False,
    **kwargs,
):
    """
    Compute the cost of an assignment over a set of constraints.

    Parameters
    ----------
    assignment: Dict[str, Any]
        The assignment given as a dict of variable_name : value
    constraints: Iterable['Constraint']
        a list of constraints
    consider_variable_cost: boolean
        if we should take into account the cost embedded in the
        variable (if any)
    **kwargs: dict
        allows passing extra variable values, is only used when a variable value
        is missing from  `assignment`.

    Raises
    ------
    TypeError:
        If some variables are missing in the assignment.

    Returns
    -------
    The sum of the costs of the constraints for this assignment.

    """
    # NOTE: this method is performance-critical and has been profiled and tuned,
    # make sure to do it again if you need to change it !!
    cost = 0
    cost_vars = None
    if consider_variable_cost:
        cost_vars = set()
    for c in constraints:
        filtered_ass = {}
        for v in c.dimensions:
            v_name = v.name
            if consider_variable_cost:
                if v_name not in cost_vars:
                    cost += v.cost_for_val(assignment[v_name])
                    cost_vars.add(v_name)
            try:
                filtered_ass[v_name] = assignment[v_name]
            except KeyError:
                filtered_ass[v_name] = kwargs[v_name]

        cost += c(**filtered_ass)

    return cost


def filter_assignment_dict(assignment, target_vars):
    """
    Filter an assignment to keep only the values of the variable that are
    present in target_var.

    :param assignment: a dict { variable_name -> value}
    :param target_vars: a list of Variable objects
    :return: a dict { variable_name -> value} with only values for variables
    in target_vars
    """

    filtered_ass = {}
    target_vars_names = [v.name for v in target_vars]
    for v in assignment:
        if v in target_vars_names:
            filtered_ass[v] = assignment[v]
    return filtered_ass


def find_arg_optimal(variable, relation, mode):
    """
    Find the value in the domain of variable that yield the optimal value  on
    this relation. Optimal can be min on max depending on the value of mode.

    :param variable: the variable
    :param relation: a function or an object implementing the Relation
    protocol and depending only on the var 'variable'
    :param mode: type of optimization, 'min' or 'max'

    :return: a pair (values, rel_value) where values is a list of values from
    the variable domain that gives the best (according to mode) value for
    this relation.
    """
    if mode == "min":
        best_rel_val = get_data_type_max(DEFAULT_TYPE)
    elif mode == "max":
        best_rel_val = get_data_type_min(DEFAULT_TYPE)
    else:
        raise ValueError("Invalid optimization mode: " + mode)

    if hasattr(relation, "dimensions"):
        if len(relation.dimensions) != 1 or relation.dimensions[0] != variable:
            raise ValueError(
                "For find_arg_optimal, the relation must depend "
                "only on the given variable : {} {}".format(relation, variable)
            )
    var_val = list()
    for v in variable.domain:
        current_rel_val = relation(v)
        if (mode == "max" and best_rel_val < current_rel_val) or (
            mode == "min" and best_rel_val > current_rel_val
        ):
            best_rel_val = current_rel_val
            var_val = [v]
        elif current_rel_val == best_rel_val:
            var_val.append(v)
    return var_val, best_rel_val


def find_optimal(
    variable: Variable, assignment: Dict, constraints: Iterable[Constraint], mode: str
):
    """
    Find the best values for a set of constraints under an assignment.

    Find the best values for `variable` for the set of `constraints`, given an
    assignment for all other variables these constraints depends on.

    Parameters
    ----------
    variable: Variable
        the variable for which we want to find the best values.
    assignment: dict
        An assignment that contains a value for all other variables involved in the set
        of constraints.
    constraints: iterable of constraints
        a set of constraints
    mode: str
        `"min"` or `"max"`

    Returns
    -------
        List[Any]
            A list of values from the domain of our variable
        float
            The cost achieved with these values.
    """
    arg_best, best_cost = None, float("inf")
    if mode == "max":
        arg_best, best_cost = None, -float("inf")
    for value in variable.domain:
        assignment[variable.name] = value
        cost = assignment_cost(assignment, constraints)

        # Take into account variable cost, if any
        if hasattr(variable, "cost_for_value"):
            cost += variable.cost_for_val(value)

        if cost == best_cost:
            arg_best.append(value)
        elif (mode == "min" and cost < best_cost) or mode == "max" and cost > best_cost:
            best_cost, arg_best = cost, [value]

    return arg_best, best_cost


def optimal_cost_value(variable: Variable, mode: str):
    """
    Find the value that optimizes the integrated cost function for a variable.
    If the variable has not cost function, simply returns a random value from the domain.

    Parameters
    ----------
    variable: Variable
        a variable with an integrated cost function
    mode: str
        'min' or 'max"

    Returns
    -------
    value:
        the value that optimizes the integrated cost function
    cost:
        the cost associated with the returned value

    """
    if hasattr(variable, "cost_for_val"):
        opt_func = min if mode == "min" else max
        best_cost, best_value = opt_func(
            (variable.cost_for_val(value), value) for value in variable.domain
        )
    else:
        best_value, best_cost = random.choice(variable.domain), None

    return best_value, best_cost


def join(u1: Constraint, u2: Constraint) -> Constraint:
    """
    Build a new Constraint by joining the two Constraint u1 and u2.

    The dimension of the new Constraint is the union of the dimensions of u1
    and u2. For any complete assignment, the value of this new relation is the sum of
    the values from u1 and u2 for the subset of this assignment that apply to
    their respective dimension.

    For more details, see the definition of the join operator in Petcu's Phd thesis.

    Dimension order is important for some operations, variables for u1 are
    listed first, followed by variables from u2 that where already used by u1
    (in the order in which they appear in u2.dimension).
    Note that relying on dimension order is fragile and discouraged,
    use keyword arguments whenever possible instead !

    Parameters
    ----------
    u1: Constraint
        n-ary relation
    u2: Constraint
        n-ary relation

    Returns
    -------
    Constraint:
        a new Constraint
    """
    dims = u1.dimensions[:]
    for d2 in u2.dimensions:
        if d2 not in dims:
            dims.append(d2)

    u_j = NAryMatrixRelation(dims, name="joined_utils")
    for ass in generate_assignment_as_dict(dims):

        u1_ass = filter_assignment_dict(ass, u1.dimensions)
        u2_ass = filter_assignment_dict(ass, u2.dimensions)
        s = u1(**u1_ass) + u2(**u2_ass)
        u_j = u_j.set_value_for_assignment(ass, s)

    return u_j


def projection(a_rel: Constraint, a_var: Variable, mode="max") -> Constraint:
    """
    The projection of a relation `a_rel` along the variable `a_var` is the
    optimization of the matrix along the axis of this variable.

    The result of `projection(a_rel, a_var)` is also a relation, with one less
    dimension than a_rel (the a_var dimension).
    For each possible instantiation of the variable other than a_var,
    the optimal instantiation for a_var is chosen and the corresponding
    utility recorded in projection(a_rel, a_var)

    Also see definition in Petcu 2007.

    Parameters
    ----------
    a_rel: Constraint
        the projected relation
    a_var: Variable
        the variable over which to project
    mode: mode as str
        'max (default) for maximization, 'min' for minimization.

    Returns
    -------
    Constraint:
        the new relation resulting from the projection
    """

    remaining_vars = a_rel.dimensions.copy()
    remaining_vars.remove(a_var)

    # the new relation resulting from the projection
    proj_rel = NAryMatrixRelation(remaining_vars)

    for partial in generate_assignment_as_dict(remaining_vars):

        _, rel_val = find_arg_optimal(a_var, a_rel.slice(partial), mode)
        proj_rel = proj_rel.set_value_for_assignment(partial, rel_val)

    return proj_rel
