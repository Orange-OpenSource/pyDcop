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
from typing import Iterable, Optional

from pydcop.computations_graph.objects import ComputationNode, ComputationGraph, Link
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import Constraint, find_dependent_relations
from pydcop.utils.simple_repr import from_repr, simple_repr


class VariableComputationNode(ComputationNode):
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

    def get_previous(self):
        for l in self.links:
            if l.type == "previous":
                return l.target
        return None

    def get_next(self):
        for l in self.links:
            if l.type == "next":
                return l.target
        return None

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
    def __init__(self, name: str, nodes: Iterable[str]) -> None:
        super().__init__(nodes, link_type="constraint_link")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return "ConstraintLink({})".format(self._name)

    def __repr__(self):
        return "ConstraintLink({}, {})".format(self._name, self.nodes)

    def __eq__(self, other):
        if super().__eq__(other) and self.name == other.name:
            return True
        return False

    def __hash__(self):
        return hash((self.type, self.nodes))


class OrderLink(Link):
    def __init__(self, link_type: str, link_source: str, link_target) -> None:
        super().__init__(link_type=link_type, nodes=[link_source, link_target])
        if link_type not in ["previous", "next"]:
            raise ValueError(
                f"Invalid link type in OrderedGraph : {link_type} "
                f"between {link_source} and {link_target}"
                f"Supported types are 'previous','next'"
            )
        self._source = link_source
        self._target = link_target

    @property
    def source(self) -> str:
        """ The source of the link.

        Returns
        -------
        str
            The name of source PseudoTreeNode computation node.
        """
        return self._source

    @property
    def target(self):
        """ The target of the link.

        Returns
        -------
        str
            The name of target PseudoTreeNode computation node.
        """
        return self._target

    def _simple_repr(self):
        r = {
            "__module__": self.__module__,
            "__qualname__": self.__class__.__qualname__,
            "type": self.type,
            "source": simple_repr(self.source),
            "target": simple_repr(self.target),
        }
        return r

    @classmethod
    def _from_repr(cls, r):
        return OrderLink(r["type"], from_repr(r["source"]), from_repr(r["target"]))


class OrderedConstraintGraph(ComputationGraph):
    def __init__(self, nodes: Iterable[VariableComputationNode]) -> None:
        super().__init__(nodes=nodes, graph_type="ConstraintHyperGraph")

        # Add order links
        sorted_nodes = sorted(self.nodes, key=lambda n: n.name)

        for n1, n2 in zip(sorted_nodes[:-1], sorted_nodes[1:]):
            # n1 next is n2
            n1.links.append(OrderLink("next", n1.name, n2.name))
            # n2 prev is n1
            n2.links.append(OrderLink("previous", n2.name, n1.name))


def build_computation_graph(
    dcop: Optional[DCOP] = None,
    variables: Iterable[Variable] = None,
    constraints: Iterable[Constraint] = None,
) -> OrderedConstraintGraph:
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

    return OrderedConstraintGraph(computations)
