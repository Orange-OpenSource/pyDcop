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

from pydcop.utils.simple_repr import SimpleRepr


class ComputationNode(SimpleRepr):
    """
    A computation node represents one computation in a computation graph.
    It is not an implementation of an actual computation but contains all
    the information needed to instantiate a concrete computation.

    `ComputationNode` is the base class that be used for all concrete
    computation graph models.

    Notes
    -----
    It must be possible to transfer (e.g. network) the definitions of a
    computation node (a serialized `ComputationNode` instance) and all edges
    (a serialized `Link` instances) where it is the source to a remote agent
    where the actual computation will be instantiated. This means a a
    concrete computation node instance must be serializable (using the
    `SimpleRepr` mixin) and must include all information required for the
    computation (e.g. Variable and associated domain, Relation for the
    constraints , etc.).

    Any subclass must also implement `__hash__()` and `__eq__()`.

    Parameters
    ----------
    name: str
        the name of the computation node, usually the name of the variable (
        or constraint) the computation is responsible for.
    node_type: str
        type of the node, usefull when the computation has several type of
        nodes (e.g. factor graphs)
    neighbors: Iterable of neighbors' names, as string
        The name of of the neighbors node. The neighbors can also be given
        with the `links` argument, but you cannot use `links` and `neighbors`
        arguments simultaneously.
    links: Iterable of Links
        Link objects pointing to the neighbors in the graph. For complex
        graph it can necessary to give links instead of neighbors names,
        as it allow attaching a type to the link (like pseudo child in
        pseudo-Tree graph) or use hyper-link.

    """
    def __init__(self, name: str, node_type: str=None,
                 links: Iterable['Link']=None,
                 neighbors: Iterable[str]=None)-> None:
        self._name = name
        self._node_type = node_type

        if links is not None and neighbors is not None:
            # If both neighbors and link are given
            raise ValueError('ComputationNode supports giving neighbors or '
                             'links but not both ')
        elif neighbors is not None:
            self._neighbors = neighbors
            self._links = [Link([name, n]) for n in self.neighbors]
        elif links is not None:
            self._links = list(links)
            self._neighbors = list(set(n for l in links for n in l.nodes
                                       if n != self._name))
        else:
            self._links = []
            self._neighbors = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._node_type

    @property
    def neighbors(self) -> Iterable[str]:
        return self._neighbors

    @property
    def links(self) ->Iterable['Link']:
        return self._links

    def __eq__(self, other):
        return self.name == other.name and self.type == other.type

    def __hash__(self):
        return hash((self._name, self._node_type))

    def __repr__(self):
        if self.type is not None:
            return 'ComputationNode({}, {})'.format(self._name, self.type)
        else:
            return 'ComputationNode({})'.format(self._name)

    def _simple_repr(self):
        # neighbors and links are two representation of the same thing,
        # only keep links in the simple repr
        r = super()._simple_repr()
        if 'neighbors' in r:
            r.pop('neighbors')
        return r


class Link(SimpleRepr):
    """
    A Computation link represent an edge, in a computation graph, from one 
    computation node to another.

    To accommodate various type of graphs, the `Link` base class models an
    hyper-edge (i.e. an edge which can have more that two vertices) and
    has a type attribute, which can be used to represent different kinds of
    relation between the nodes involved in the edge (e.g. parent, children,
    pseudo children, neighbor, ...)

    When necessary, specific graph model can extend this base class to define
    other types of edges : binary, directed, etc.

    In any case, all edge object mus be serializable, using the `SimpleRepr`
    mixin, and must implement `__hash__()` and `__eq__()`.
    """

    def __init__(self, nodes: Iterable[str], link_type: str = None)-> None:
        """

        :param link_type: type of link
        :param nodes: iterable of computation names
        """
        self._link_type = link_type
        self._nodes = frozenset(nodes)

    @property
    def type(self):
        return self._link_type

    @property
    def nodes(self) -> Iterable[str]:
        return self._nodes

    def has_node(self, node_name: str) -> bool:
        return node_name in self._nodes

    def __str__(self):
        if self.type is not None:
            return 'Link({}, {})'.format(self.type, self.nodes)
        else:
            return 'Link({})'.format(self.nodes)

    def __repr__(self):
        if self.type is not None:
            return 'Link({}, {})'.format(self.type, self.nodes)
        else:
            return 'Link({})'.format(self.nodes)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.type == other.type and self.nodes == other.nodes:
            return True
        return False

    def __hash__(self):
        return hash((self.type, self.nodes))


class ComputationGraph(object):
    """
    A ComputationGraph represents a graph of computation for a dcop.
    Many different graph can be defined from the same dcop, depending on the 
    kind of algorithm that will be used to solve the dcop. For example, 
    max-sum is based on a factor graph while dpop is based on a DFS 
    pseudo-tree.
    For this reason, we try to keep the computation graph definition both 
    flexible and generic, so that it can accommodate and represents many  
    kind of graph models.

    Concrete sub-classes MUST provide `nodes` and `links` attributes (or
    properties) that return respectively the list of `ComputationNode`s and
    the list of  `Link`s of the graph

    Parameters
    ----------
    graph_type:
    nodes: ComputationNodes
        an iterable of ComputationNode objects of None

    Examples
    --------
    >>> cg = ComputationGraph()
    >>> cg.nodes
    []

    >>> cg = ComputationGraph(nodes=[ComputationNode('a1'),
    ...                              ComputationNode('a2')])
    >>> cg.nodes
    [ComputationNode(a1), ComputationNode(a2)]
    """

    def __init__(self, graph_type: str=None,
                 nodes: Iterable[ComputationNode]=None)-> None:
        self.type = graph_type
        self.nodes = [] if nodes is None else list(nodes)

    @property
    def links(self):
        links = set()
        for n in self.nodes:
            links.update(l for l in n.links)
        return links

    def node_names(self):
        return [n.name for n in self.nodes]

    def computation(self, node_name: str)-> ComputationNode:
        """Return a computation node from its name.

        Parameters
        ----------
        node_name: str
            a computation node name

        Returns
        -------
        A ComputationNode object with this name.

        Raises
        ------
        A KeyError if no computation with this name could be found

        Examples
        --------
        >>> cg = ComputationGraph()
        >>> cg.nodes = [ComputationNode('a1'), ComputationNode('a2')]
        >>> cg.computation('a1')
        ComputationNode(a1)

        """
        for n in self.nodes:
            if node_name == n.name:
                return n
        raise KeyError('no computation named {} found'.format(node_name))

    def links_for_node(self, node_name: str) -> Iterable[Link]:
        """Return the links involving a given computation.

        Parameters
        ----------
        node_name: str
            a computation node name

        Returns
        -------
        An iterable of Link objects involving the computation

        Examples
        --------
        >>> cg = ComputationGraph(
        ...         nodes= [ComputationNode('a1', neighbors=['a2']),
        ...                 ComputationNode('a2', neighbors=['a1'])])
        >>> Link({'a1', 'a2'}) in cg.links_for_node('a1')
        True
        """
        for n in self.nodes:
            if n.name == node_name:
                return n.links
        raise KeyError('No node named '+node_name)

    def neighbors(self, node_name: str) -> Iterable[str]:
        """Return the neighbors of a computation node.

        Iterates over names of neighbors instead of ComputationNode objects.

        Parameters
        ----------
        node_name: string
            a computation node name.

        Returns
        -------
        An iterable over the names of the neighbors of the computation in the
        computation graph.

        Examples
        --------
        >>> cg = ComputationGraph(
        ...         nodes= [ComputationNode('a1', neighbors=['a2']),
        ...                 ComputationNode('a2', neighbors=['a1'])])
        >>> list(cg.neighbors('a1'))
        ['a2']

        """
        for n in self.nodes:
            if n.name == node_name:
                return n.neighbors
        raise KeyError('No node named '+node_name)

    def density(self):
        raise NotImplementedError('Abstract class')
