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


import pkgutil
from importlib import import_module
from typing import List

import numpy as np

from pydcop.algorithms.objects import AlgoDef
from pydcop.computations_graph.objects import ComputationNode
from pydcop.dcop.objects import Variable
from pydcop.utils.simple_repr import SimpleRepr


def list_available_algorithms():
    exclude_list = {'generic_computations', 'graphs', 'objects'}
    algorithms = []

    root_algo = import_module('pydcop.algorithms')
    for importer, modname, ispkg in pkgutil.iter_modules(root_algo.__path__,
                                                         ''):
        if modname not in exclude_list:
            algorithms.append(modname)

    return algorithms


DEFAULT_TYPE = np.int32


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
    if mode == 'min':
        best_rel_val = get_data_type_max(DEFAULT_TYPE)
    elif mode == 'max':
        best_rel_val = get_data_type_min(DEFAULT_TYPE)
    else:
        raise ValueError('Invalid optimization mode: ' + mode)

    if hasattr(relation, 'dimensions'):
        if len(relation.dimensions) != 1 or relation.dimensions[0] != variable:
            raise ValueError('For find_arg_optimal, the relation must depend '
                             'only on the given variable : {} {}'
                             .format(relation, variable))
    var_val = list()
    for v in variable.domain:
        current_rel_val = relation(v)
        if (mode == 'max' and best_rel_val < current_rel_val) or \
                (mode == 'min' and best_rel_val > current_rel_val):
            best_rel_val = current_rel_val
            var_val = [v]
        elif current_rel_val == best_rel_val:
            var_val.append(v)
    return var_val, best_rel_val


ALGO_STOP = 0
ALGO_CONTINUE = 1
ALGO_NO_STOP_CONDITION = 2


class ComputationDef(SimpleRepr):
    """
    A Computation node contains all the information needed to create a
    computation instance that can be run. It can be used when deploying the
    computation or as a replica when distributing copies of a computation for
    resilience.
    """

    def __init__(self, node: ComputationNode, algo: AlgoDef) -> None:
        self._node = node
        self._algo = algo

    @property
    def algo(self) -> AlgoDef:
        return self._algo

    @property
    def node(self) -> ComputationNode:
        return self._node

    @property
    def name(self):
        return self.node.name

    def __str__(self):
        return 'ComputationDef({}, {})'.format(self.node.name, self.algo.algo)

    def __repr__(self):
        return 'ComputationDef({}, {})'.format(self.node, self.algo)

    def __eq__(self, other):
        if type(other) != ComputationDef:
            return False
        if self.node == other.node and self.algo == other.algo:
            return True
        return False
