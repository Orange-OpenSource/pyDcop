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


import random
import networkx as nx


class Node(object):
    """
    A generic Node for a bipartite graph
    """

    def __init__(self, content, node_type=None):
        """

        :param node_type: Type of node (for bi partite nodes)
        :param content: the content object must have a name property that
        will be used as the node name (and id, which means that in a graph
        all names must be different)
        """
        self.content = content
        self.type = node_type
        self.neighbors = []

    @property
    def name(self):
        return self.content.name

    def add_neighbors(self, node, directed=False):
        if node.type is not None and self.type == node.type:
            raise ValueError(
                "In a bipartite graph two nodes with the same "
                "type cannot be connected : {} - {}".format(node, self)
            )
        self.neighbors.append(node)
        if not directed:
            node.add_neighbors(self, directed=True)


def as_bipartite_graph(variables, relations):
    nodes = {}

    for v in variables:
        n = Node(v, "VARIABLE")
        nodes[v.name] = n

    for r in relations:
        n = Node(r, "CONSTRAINT")
        nodes[r.name] = n
        for v in r.dimensions:
            current_var_neighbors = [n.content for n in nodes[v.name].neighbors]
            if v not in current_var_neighbors:
                n.add_neighbors(nodes[v.name])

    return nodes.values()


def calc_diameter(nodes):
    """
    Warning : this only works on tree graphs !!

    For arbitrary graphs, we need to compute the shortest path between any
    two vertices and take the length of the greatest of these paths
    :param nodes:
    :return:
    """

    # Calculate the diameter of a graph made of variables and relations

    # first pick a random node in the tree and use a BFS to find the furthest
    # node in the graph
    root = random.choice(nodes)
    node, distance = find_furthest_node(root, nodes)

    _, distance = find_furthest_node(node, nodes)

    return distance


def find_furthest_node(root_node, nodes):

    # BFS on the graph defined by nodes
    queue = [root_node]
    distances = {root_node.name: 0}
    max_distance = 0
    furthest_node = root_node
    while len(queue) > 0:
        current = queue.pop()

        for neighbor in current.neighbors:
            d = distances.get(neighbor.name, -1)
            if d == -1:
                d = distances[current.name] + 1
                if d > max_distance:
                    max_distance = d
                    furthest_node = neighbor
                distances[neighbor.name] = d
                queue.append(neighbor)

    return furthest_node, max_distance


def as_networkx_graph(variables, relations):
    """
    Build a networkx graph object from variables and relations.

    Parameters
    ----------
    variables: list
        a list of Variable objets
    relations: list
        a list of Relation objects

    Returns
    -------
    a networkx graph object
    """
    graph = nx.Graph()

    # One node for each variables
    graph.add_nodes_from([v.name for v in variables])

    for r in relations:
        for p in all_pairs([e.name for e in r.dimensions]):
            graph.add_edge(*p)
    return graph


def as_networkx_bipartite_graph(variables, relations):
    """
    Build a networkx graph object from variables and relations.

    Parameters
    ----------
    variables: list
        a list of Variable objets
    relations: list
        a list of Relation objects

    Returns
    -------
    a networkx graph object
    """
    graph = nx.Graph()

    # One node for each variables
    graph.add_nodes_from([v.name for v in variables], bipartite=0)
    graph.add_nodes_from([r.name for r in relations], bipartite=1)

    for r in relations:
        for e in r.dimensions:
            graph.add_edge(r.name, e.name)
    return graph


def display_graph(variables, relations):
    """
    Display the variables and relation as a graph, using networkx and
    matplotlib.

    Parameters
    ----------

    variables: list
        a list of Variable objets
    relations: list
        a list of Relation objects
    """
    graph = as_networkx_graph(variables, relations)

    # Do not crash if matplotlib is not installed
    try:
        import matplotlib.pyplot as plt

        nx.draw_networkx(graph, with_labels=True)
        # nx.draw_random(graph)
        # nx.draw_circular(graph)
        # nx.draw_spectral(graph)
        plt.show()
    except ImportError:
        print("ERROR: cannot display graph, matplotlib is not installed")


def display_bipartite_graph(variables, relations):
    """
    Display the variables and relation as a graph, using networkx and
    matplotlib.

    Parameters
    ----------
    variables: list
        a list of Variable objets
    relations: list
        a list of Relation objects
    """
    graph = as_networkx_bipartite_graph(variables, relations)

    # Do not crash if matplotlib is not installed
    try:
        import matplotlib.pyplot as plt

        pos = nx.drawing.spring_layout(graph)
        variables = set(n for n, d in graph.nodes(data=True) if d["bipartite"] == 0)
        factors = set(graph) - variables
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            with_labels=True,
            nodelist=variables,
            node_shape="o",
            node_color="b",
            label="variables",
            alpha=0.5,
        )
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            with_labels=True,
            nodelist=factors,
            node_shape="s",
            node_color="r",
            label="factors",
            alpha=0.5,
        )
        nx.draw_networkx_labels(graph, pos=pos)
        nx.draw_networkx_edges(graph, pos=pos)
        # nx.draw_random(graph)
        # nx.draw_circular(graph)
        # nx.draw_spectral(graph)
        plt.show()
    except ImportError:
        print("ERROR: cannot display graph, matplotlib is not installed")


def cycles_count(variables, relations):

    g = as_networkx_graph(variables, relations)
    cycles = nx.cycle_basis(g)
    return len(cycles)


def graph_diameter(variables, relations):
    """
    Compute the graph diameter(s).
    If the graph contains several independent sub graph, returns a list the
    diamater of each of the subgraphs.

    :param variables:
    :param relations:
    :return:
    """
    diams = []
    g = as_networkx_graph(variables, relations)
    components  = (g.subgraph(c).copy() for c in nx.connected_components(g))
    for c in components:
        diams.append(nx.diameter(c))

    return diams


def all_pairs(elements):
    """
    Generate all possible pairs from the list of given elements.

    Pairs have no order: (a, b) is the same as (b, a)

    :param elements: an array of elements
    :return: a list of pairs, for example [('a', 'b)]
    """
    if len(elements) < 2:
        return []
    elif len(elements) == 2:
        return [(elements[0], elements[1])]
    else:
        new_pairs = []
        for elt in elements[1:]:
            new_pairs.append((elements[0], elt))
        return all_pairs(elements[1:]) + new_pairs
