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


from typing import Iterable

from pydcop.computations_graph.objects import ComputationNode, ComputationGraph, Link
from pydcop.dcop.dcop import DCOP, Variable
from pydcop.dcop.relations import find_dependent_relations, Constraint

"""
This module implements the classical constraint graph model.

In this graph model,

*  each variables is represented by one node in the graph

* edges 
"""


class VariableComputationNode(ComputationNode):
    """A VariableComputationNode represent a computation responsible for
    selecting the value of one variable, in a computation constrains
    hyper-graph.

    Parameters
    ----------
    variable: Variable
        The variable this computation is responsible for.
    constraints: Iterable of constraints
        The Constraints the variable depends on
    name: str
        The name of the node. If given given, the name of the variable is
        used as the node name.

    See Also
    --------
    ComputationConstraintsHyperGraph

    """

    def __init__(
        self, variable: Variable, constraints: Iterable[Constraint], name: str = None
    ) -> None:
        if name is None:
            name = variable.name
        links = []
        for c in constraints:
            links.append(
                ConstraintLink(name=c.name, nodes=[v.name for v in c.dimensions])
            )
        super().__init__(name, "VariableComputationNode", links=links)
        self._variable = variable
        self._constraints = constraints

    @property
    def variable(self):
        return self._variable

    @property
    def constraints(self):
        return self._constraints

    def __eq__(self, other):
        if type(other) != VariableComputationNode:
            return False
        if self.variable == other.variable and self.constraints == other.constraints:
            return True
        return False

    def __str__(self):
        return "VariableComputationNode({})".format(self._variable.name)

    def __repr__(self):
        return "VariableComputationNode({}, {})".format(
            self._variable, self.constraints
        )

    def __hash__(self):
        return hash(
            (self._name, self._node_type, self.variable, tuple(self.constraints))
        )


class ConstraintLink(Link):
    """Link between nodes in a constraint hyper-graph

    Parameters
    ----------
    name: str
        the name of the constraint represented by this edge.
    nodes: Iterable of nodes names, as str
        the names of the VariableComputationNode corresponding to the
        variables in the scope of the constraint represented by the edge.

    """

    def __init__(self, name: str, nodes: Iterable[str]) -> None:
        super().__init__(nodes, link_type="constraint_link")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return "ConstraintGraphLink({})".format(self._name)

    def __repr__(self):
        return "ConstraintGraphLink({}, {})".format(self._name, self.nodes)

    def __eq__(self, other):
        if super().__eq__(other) and self.name == other.name:
            return True
        return False

    def __hash__(self):
        return hash((self.type, self.nodes))


class ComputationConstraintsHyperGraph(ComputationGraph):

    """
    A `ComputationConstraintsHyperGraph` represents computation graph that
    have the same extract structure as the constraint hyper of the
    underlying optimization problem. The vertices of the graph represents
    the decisions variable and edges represents constraints.

    This type of graph is used with algorithms that work directly on the
    constraints hyper graph (without any preprocessing step) and define one
    computation for each decision variable (e.g. DSA, MGM, etc).

    """

    def __init__(self, nodes: Iterable[VariableComputationNode]) -> None:
        super().__init__(nodes=nodes, graph_type="ConstraintHyperGraph")

    def density(self) -> float:
        # Whats the correct definition of density for hypergraphs ?
        # propably not, the maximum number of edges in an hypergraph is the
        # sum of number of possible set with 1, 2 , 3 .... n edges.
        # = 2^n +1 ?
        e = len(self.links)
        v = len(self.nodes)
        return 2 * e / (v * (v - 1))


def build_computation_graph(
    dcop: DCOP = None,
    variables: Iterable[Variable] = None,
    constraints: Iterable[Constraint] = None,
) -> ComputationConstraintsHyperGraph:
    """
    Build a computation hyper graph for the DCOP.

    A computation graph is generally built from a DCOP, however it is also
    possible to build a sub-graph of the computation graph by simply passing
    the variables and constraints.

    Parameters
    ----------
    dcop: DCOP
        DCOP object to build the computation graph from.When this
        parameter is used, the `constraints` and `variables` parameters MUST
        NOT be used.
    variables: iterable of Variables objects
        The variables to build the computation graph from. When this
        parameter is used, the `constraints` parameter MUST also be given.
    constraints: iterable of Constraints objects
        The constraints to build the computation graph from. When this
        parameter is used, the `variables` parameter MUST also be given.

    Returns
    -------
    ComputationConstraintsHyperGraph
        In hyper-graph for the variables and constraints

    Raises
    ------
    ValueError
        If both `dcop` and one of the `variables` or `constraints` arguments
        have been used.

    """
    computations = []
    if dcop is not None:
        if constraints or variables is not None:
            raise ValueError(
                "Cannot use both dcop and constraints / " "variables parameters"
            )
        for v in dcop.variables.values():
            var_constraints = find_dependent_relations(v, dcop.constraints.values())
            computations.append(VariableComputationNode(v, var_constraints))
    else:
        if constraints is None or variables is None:
            raise ValueError(
                "Constraints AND variables parameters must be "
                "provided when not building the graph from a dcop"
            )
        for v in variables:
            var_constraints = find_dependent_relations(v, constraints)
            computations.append(VariableComputationNode(v, var_constraints))

    # links = []
    # for r in dcop.constraints.values():
    #     in_scope = (v.name for v in r.dimensions)
    #     links.append(ConstraintLink(r.name, in_scope))

    return ComputationConstraintsHyperGraph(computations)
