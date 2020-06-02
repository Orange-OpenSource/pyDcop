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


"""
Pseudo tree computation graphs are DFS trees built from the constraint graph 
where pseudo-parent and pseudo-children links are added to the tree.
 
 This model is typically used for the dpop algorithm.
"""
from typing import Dict
from typing import Iterable

from collections import defaultdict
from typing import List

from pydcop.computations_graph.objects import ComputationNode, ComputationGraph, Link
from pydcop.dcop.objects import Variable
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import RelationProtocol, Constraint
from pydcop.utils.simple_repr import from_repr, simple_repr


class PseudoTreeLink(Link):
    """
    Link between to node in a pseudo-tree computation graph.


    """

    def __init__(self, link_type: str, source: str, target: str) -> None:
        """

        Parameters
        ----------
        link_type: str
            pseudo-tree graph uses 3 types of links: 'children',
            'pseudo_children', 'parent' and 'pseudo_parent'. Raise a Value error
            if the link_type is not valid.

        source: str
            The source of the link, it must be the name of a PseudoTreeNode
            in the graph.
        target: str
            The target of the link, it must be the name of a PseudoTreeNode
            in the graph.
        """
        if link_type not in ["children", "pseudo_children", "pseudo_parent", "parent"]:
            raise ValueError(
                "Invalid link type in pseudo-tree "
                'graph: {}. Supported types are "children",'
                '"pseudo_children" and "pseudo_parent"'.format(link_type)
            )
        super().__init__(link_type=link_type, nodes=[source, target])
        self._source = source
        self._target = target

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
        return PseudoTreeLink(r["type"], from_repr(r["source"]), from_repr(r["target"]))


class PseudoTreeNode(ComputationNode):
    """ComputationNode for Pseudo-tree computation graph.

    Parameters
    ----------
    variable: Variable
        The variable this computation is responsible for.
    constraints: iterable of constraints-like objects
        The constraints the variable depends on.
    links: iterable of PseudoTreeLink
        The links are mandatory because we cannot extract them from the
        variable and constraints of this node (as it is the case for factor
        graph or hyper-graph), they depends on the DFS pseudo tree generation.
    name: str
        The name of the node. If given given, the name of the variable is
        used as the node name.

    """

    def __init__(
        self,
        variable: Variable,
        constraints: Iterable[Constraint],
        links: Iterable[PseudoTreeLink],
        name: str = None,
    ) -> None:
        name = name if name is not None else variable.name
        super().__init__(name, "PseudoTreeComputation", links=links)
        self._variable = variable
        self._constraints = tuple(constraints)

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def constraints(self) -> Iterable[RelationProtocol]:
        return self._constraints

    def __str__(self):
        return "PseudoTreeNode({},{})".format(self._variable, self._constraints)

    def __repr__(self):
        return "PseudoTreeNode({},{})".format(self._variable, self._constraints)

    def __eq__(self, other):
        if type(other) != PseudoTreeNode:
            return False
        if self.variable == other.variable and self.constraints == other.constraints:
            return True
        return False

    def __hash__(self):
        return hash((self._variable, self._constraints))


def get_dfs_relations(tree_node: PseudoTreeNode):
    """
    Utility function to get lists of descendants and ancestors.

    Parameters
    ----------
    tree_node: PseudoTreeNode
        a node in a dfs tree

    Returns
    -------
    tuple:
        a tuple (parent, pseudo_parents, children, pseudo_children)
    """
    parent = None
    pseudo_parents = []
    children = []
    pseudo_children = []

    for l in tree_node.links:
        if l.type == "parent" and l.source == tree_node.name:
            parent = l.target
        if l.type == "children" and l.source == tree_node.name:
            children.append(l.target)
        if l.type == "pseudo_children" and l.source == tree_node.name:
            pseudo_children.append(l.target)
        if l.type == "pseudo_parent" and l.source == tree_node.name:
            pseudo_parents.append(l.target)

    return parent, pseudo_parents, children, pseudo_children


class _BuildingNode(object):
    """
    This class is only used when building the pseudo tree and should never be
    used outside this module.

     It should probably be refactored ...
    """

    def __init__(self, variable):
        # super().__init__(variable.name, 'PseudoTreeComputation')
        self._variable = variable
        self._neighbors = []
        self.relations = []
        self.parent = None
        self.pseudo_parents = []
        self.pseudo_children = []
        self.children = []
        self._visited = []
        self.root = False

    @property
    def name(self):
        return self._variable.name

    @property
    def variables(self):
        return [self._variable]

    @property
    def variable(self):
        return self._variable

    def handle_token(self, sender, token):
        token = token[:]
        self._visited.append(sender)
        if sender is None:
            # root
            self.root = True
            self._propagate(token)

        elif self.parent is None and not self.root:
            self.parent = sender
            self.pseudo_parents = [
                n for n in self._neighbors if n in token and n != sender
            ]
            self._neighbors.sort(
                key=lambda x: x.count_neighbors_in_token(token), reverse=True
            )
            self._propagate(token)

        else:
            if sender in self.children:
                pass
            else:
                self.pseudo_children.append(sender)

    def _propagate(self, token):
        token.append(self)

        # heuristic :
        # sort our neighbors based on the number of their neighbors are
        # already in the token
        self._neighbors.sort(
            key=lambda x: x.count_neighbors_in_token(token), reverse=True
        )

        for n in self._neighbors:
            if n not in self._visited:
                if n not in self.pseudo_parents:
                    self.children.append(n)
                n.handle_token(self, token)

    def count_neighbors_in_token(self, token):
        """
        Count the number of our neighbors that are in the token.
        Used for DFS heuristic
        """
        c = 0
        for n in self._neighbors:
            if n in token:
                c += 1
        return c

    def neighbors_count(self):
        return len(self._neighbors)

    def __repr__(self):
        return "Node " + self.variable.name

    def __str__(self):
        return "Node " + self.variable.name


def _find_neighbors_relations(node, relations, nodes):
    """
    Find all neighbors and relation for this node.

    :param node: the node we search the neighbors and relations for
    :param relations: a list of all relations
    :param nodes: a list of all nodes
    :return: a pair (neighbors, relations)
    """
    node_neighbors = []
    node_relations = []
    for r in relations:
        if node.variable in r.dimensions:
            node_relations.append(r)
            dim_vars = list(r.dimensions)
            dim_vars.remove(node.variable)
            for n in nodes:
                if n.variable in dim_vars and n not in node_neighbors:
                    node_neighbors.append(n)
    return node_neighbors, node_relations


def _generate_dfs_tree(variables, relations, root=None):
    """
    Generate a DFS tree for these variables connected by these relations.
    If the 'root' is argument is not None, it is used as the root of the
    tree, otherwise the node with the highest number of neighbors is used as root.


    :param variables:
    :param relations:
    :param root:
    :return: the root of the pseudo-tree
    """

    # build a node for each of the variables
    nodes = []
    for v in variables:
        n = _BuildingNode(v)
        nodes.append(n)
    for n in nodes:
        neighbors, rels = _find_neighbors_relations(n, relations, nodes)
        n._neighbors = neighbors
        n.relations = rels

    # Root selection with heuristic : choose the Node with the highest number
    #  of neighbors
    if root is None:
        # heuristic :
        # root = random.choice(variables)
        nodes.sort(key=lambda n: n.neighbors_count())
        root = nodes[-1]
    else:
        for n in nodes:
            if n.variable == root:
                root = n
                break

    token = []
    root.handle_token(None, token)

    return root


def _visit_tree(root):
    """
    Iterator: visit a tree, yielding each node in DFS order.

    :param root: the root node of the tree.
    """
    yield root
    for c in root.children:
        # Using 'yield from would be nicer, but is only available with python
        #  >= 3.3
        for n in _visit_tree(c):
            yield n


def tree_str_desc(root, indent_num=0):
    """
    Build a string representing a pseudo-tree
    :param root:
    :param indent_num:
    :return:
    """
    desc = ""
    indent = " " * indent_num
    pp = ", ".join([p.variable.name for p in root.pseudo_parents])
    pc = ", ".join([c.variable.name for c in root.pseudo_children])
    desc += (
        indent + "* " + root.variable.name + " - PP : [" + pp + "] - PC: [" + pc + "]\n"
    )
    for n in root.children:
        desc += tree_str_desc(n, indent_num=(indent_num + 2))
    return desc


class ComputationPseudoTree(ComputationGraph):
    """Pseudo-tree based computation graph.

    The kind of computation graph is used by algorithm that need to build a
    pseudo-tree from the constraints graph.

    """

    def __init__(self, roots: Iterable[_BuildingNode]) -> None:
        super().__init__("PseudoTree")
        self._roots = list(roots)

        # build the list of links
        links = defaultdict(lambda: [])  # type: Dict[str, List]
        _nodes = {}
        for root in self._roots:
            for n in _visit_tree(root):
                if n.parent is not None:
                    links[n.name].append(
                        PseudoTreeLink("parent", n.name, n.parent.name)
                    )
                for c in n.children:
                    links[n.name].append(PseudoTreeLink("children", n.name, c.name))
                for c in n.pseudo_children:
                    links[n.name].append(
                        PseudoTreeLink("pseudo_children", n.name, c.name)
                    )
                for c in n.pseudo_parents:
                    links[n.name].append(
                        PseudoTreeLink("pseudo_parent", n.name, c.name)
                    )

            for n in _visit_tree(root):
                _nodes[n.name] = PseudoTreeNode(n.variable, n.relations, links[n.name])

        self.nodes = list(_nodes.values())

    @property
    def roots(self):
        return self._roots

    def density(self):
        # pseudo tree are directed graph, so density is e / (v - (v - 1)).
        e = len(self.links)
        v = len(self.nodes)
        return e / (v * (v - 1))

    def __str__(self):
        return f"PseudoTree nodes={ [n.name for n in self.nodes]} " \
               f"links={[l for l in self.links]}"


def _filter_relation_to_lowest_node(dfs_root):
    """"
    Filter the relations on all the nodes of the DFS tree to only keep the
    relation on the on the lowest node in the tree that is involved in the
    relation.

    """
    for n in _visit_tree(dfs_root):

        keep_rel = n.relations[:]
        for r in n.relations:
            # filter out all relations that depends on one of our children or
            #  pseudo children
            for pc in n.pseudo_children + n.children:
                if pc.variable in r.dimensions:
                    keep_rel.remove(r)
                    break
        n.relations = keep_rel


def build_computation_graph(
    dcop: DCOP,
    variables: Iterable[Variable] = None,
    constraints: Iterable[Constraint] = None,
) -> ComputationPseudoTree:
    """
    Build a computation pseudo-tree graph for the DCOP.

    A computation graph is generally built from a DCOP, however it is also
    possible to build it by simply passing the variables and constraints.

    Notes
    -----
    With DFS pseudo tree, all the DCOP is needed, you cannot build a sub-graph
    by passing a subset of the variables and constraints.

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
    ComputationPseudoTree
        In pseudo-tree graph for the variables and constraints

    Raises
    ------
    ValueError
        If both `dcop` and one of the `variables` or `constraints` arguments
        have been used.

    """

    if dcop is not None:
        if constraints or variables is not None:
            raise ValueError(
                "Cannot use both dcop and constraints / " "variables parameters"
            )
        variables = list(dcop.variables.values())
        constraints = list(dcop.constraints.values())
    else:
        if constraints is None or variables is None:
            raise ValueError(
                "Constraints AND variables parameters must be "
                "provided when not building the graph from a dcop"
            )
        variables = list(variables)
        constraints = list(constraints)

    roots = []
    while len(variables) != 0:
        root = _generate_dfs_tree(variables, constraints)
        roots.append(root)
        # Remove variables that are part of the tree and build another tree
        # until there is no variable left.
        for node in _visit_tree(root):
            variables.remove(node.variable)

    return ComputationPseudoTree(roots)
