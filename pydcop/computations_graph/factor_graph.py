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


from itertools import chain
from typing import Iterable, Union
from typing import List
from typing import Set

from pydcop.computations_graph.objects import ComputationNode, Link,\
    ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import Constraint, find_dependent_relations
from pydcop.utils.simple_repr import SimpleRepr, simple_repr, from_repr


class FactorComputationNode(ComputationNode):
    """Factor ComputationNode for factor-graph.

    A factor-graph is a bipartite graph made of two kind of nodes,
    FactorComputationNode and VariableComputationNode.

    Parameters
    ----------
    factor: Constraint
        The constraint this computation is responsible for.
    name: str
        Optional name, if not given the name of the constraint is used as the
        name of the node.

    """

    def __init__(self, factor: Constraint, name: str=None)-> None:
        name = name if name is not None else factor.name
        links = []
        for v in factor.dimensions:
            links.append(FactorGraphLink(name, v.name))
        super().__init__(name, 'FactorComputation', links=links)
        self._factor = factor
        self._variables = list(factor.dimensions)

    @property
    def constraints(self):
        return [self._factor]

    @property
    def variables(self):
        return self._variables

    @property
    def factor(self):
        return self._factor

    def __str__(self):
        return 'FactorComputationNode({})'.format(self._factor)

    def __repr__(self):
        return 'FactorComputationNode({}, {})'.format(self._factor.name,
                                                      self._variables)

    def __eq__(self, other):
        if type(other) != FactorComputationNode:
            return False
        if self.factor == other.factor and self.variables == other.variables:
            return True
        return False

    def __hash__(self):
        return hash((self._factor, tuple(self._variables)))

    def _simple_repr(self):
        r = SimpleRepr._simple_repr(self)
        return r


class VariableComputationNode(ComputationNode):
    """Variable ComputationNode for factor-graph.

    A factor-graph is a bipartite graph made of two kind of nodes,
    FactorComputationNode and VariableComputationNode.

    Parameters
    ----------
    variable: Variable
        The Variable this computation is responsible for.
    constraints_names: Iterable of str
        The name of the constraints this variable is involved in.
    name: str
        Optional name, if not given the name of the variable is used as the
        name of the node.

    """
    def __init__(self, variable: Variable,
                 constraints_names: Iterable[str],
                 name: Union[str, None]=None)-> None:
        name = name if name is not None else variable.name
        self._constraints_names = constraints_names  # type: Iterable[str]
        links = []
        for c in self._constraints_names:
            links.append(FactorGraphLink(c, name))
        super().__init__(name, 'VariableComputation', links=links)
        self._variable = variable

    @property
    def variable(self):
        return self._variable

    @property
    def constraints_names(self):
        return self._constraints_names

    def __str__(self):
        return 'VariableComputationNode({})'.format(self._variable)

    def __repr__(self):
        return 'VariableComputationNode({})'.format(self._variable)

    def __eq__(self, other):
        if type(other) != VariableComputationNode:
            return False
        if self.variable == other.variable:
            return True
        return False

    def __hash__(self):
        return hash(self._variable)

    def _simple_repr(self):
        r = SimpleRepr._simple_repr(self)
        return r


class FactorGraphLink(Link):

    """
    In Factor Graph, links are binary (FG are not hyper-graphs) and are
    between a factor node and a variable node.

    Parameters
    ----------
    factor_node: str
        The name of the factor node.
    variable_node: str
        The name of the variable node.

    """
    def __init__(self, factor_node: str,
                 variable_node: str)-> None:
        super().__init__([factor_node, variable_node],
                         link_type='fg_neighbor')
        self._factor_node = factor_node
        self._variable_node = variable_node

    @property
    def factor_node(self) -> str:
        return self._factor_node

    @property
    def variable_node(self) -> str:
        return self._variable_node

    def __str__(self):
        return 'FactorGraphLink({}, {} )'.format(self.type, self.nodes)

    def __repr__(self):
        return 'FactorGraphLink({}, {} )'.format(self.type, self.nodes)

    def _simple_repr(self):
        r = {'__module__': self.__module__,
             '__qualname__': self.__class__.__qualname__,
             'factor': simple_repr(self.factor_node),
             'variable': simple_repr(self.variable_node)
             }
        return r

    @classmethod
    def _from_repr(cls, r):
        return FactorGraphLink(from_repr(r['factor']),
                               from_repr(r['variable']))


class ComputationsFactorGraph(ComputationGraph):
    """
    A ComputationSFactorGraph is a computation graph based on a factor-graph 
    model.
    
    The graph is made of two kind of nodes : 
    * variables nodes
    * factor nodes

    Edges can only exists between a factor node and a variable node.

    """

    def __init__(self, var_nodes: Iterable[VariableComputationNode],
                 factor_nodes: Iterable[FactorComputationNode])-> None:
        # Avoid copy-paste error : ensure we do not have two computations
        nodes = list(chain(var_nodes,
                           factor_nodes))  # type: List[ComputationNode]
        # with the same name
        c_names = set()  # type: Set[str]
        for vn in nodes:
            if vn.name in c_names:
                raise KeyError('duplicate computation names: {}'.format(
                    vn.name))
            c_names.add(vn.name)
        super().__init__('FactorGraph', nodes=nodes)

    def density(self):

        # FG are undirected graphs, the density is 2 |E| / (|V| * (|V|-1)
        e = len(self.links)
        v = len(self.nodes)
        return 2 * e / (v * (v - 1))


def build_computation_graph(dcop: DCOP,
                            variables: Iterable[Variable]=None,
                            constraints: Iterable[Constraint]=None,
                            )-> ComputationsFactorGraph:
    """Build a Factor graph computation graph for a DCOP.


    Parameters
    ----------
    dcop: DCOP
        a DCOP objects containing variables and constraints
    variables: iterable of Variables objects
        The variables to build the computation graph from. When this
        parameter is used, the `constraints` parameter MUST also be given.
    constraints: iterable of Constraints objects
        The constraints to build the computation graph from. When this
        parameter is used, the `variables` parameter MUST also be given.

    """

    # TODO : external variables computation ?
    if dcop is not None:
        if constraints or variables is not None:
            raise ValueError('Cannot use both dcop and constraints / '
                             'variables parameters')
        variables = dcop.variables.values()
        constraints = dcop.constraints.values()
    elif constraints is None or variables is None:
            raise ValueError('Constraints AND variables parameters must be '
                             'provided wgen not building the graph from a dcop')

    var_nodes = []
    for v in variables:
        dep = find_dependent_relations(v, constraints)
        var_nodes.append(VariableComputationNode(
            v, constraints_names=[d.name for d in dep]))

    factor_nodes = []
    for c in constraints:
        n = FactorComputationNode(c)
        factor_nodes.append(n)

    fg = ComputationsFactorGraph(var_nodes, factor_nodes)
    return fg
