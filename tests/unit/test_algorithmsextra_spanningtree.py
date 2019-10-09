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
Tests for the GHS Distributed Minimum Spanning tree algorithm.

* TODO: test with several node waking up spontaneously
* TODO: check minimum tree with networkx - for larger graphs
* TODO: test with disconnected graph ?

"""
import random
from time import sleep
from typing import Dict

import networkx as nx
import pytest

from pydcop.algorithmsextra.spanningtree import (
    SpanningTreeComputation,
    EdgeLabel,
    inf_val,
    is_best_weight,
    find_best_edge,
    to_edge,
)
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer


def test_to_edge():
    assert to_edge("A", "B") == ("A", "B")
    assert to_edge("D", "B") == ("B", "D")


def test_inf_value():
    assert inf_val("min") == (float("inf"),)
    assert inf_val("max") == (-float("inf"),)


def test_best_val():
    assert is_best_weight(2, 5, "min")
    assert not is_best_weight(8, 5, "min")

    assert is_best_weight(7, 1, "max")
    assert not is_best_weight(2, 5, "max")


def test_find_best_edge():

    assert find_best_edge({"A": 5, "B": 1, "C": 3}, "min") == ("B", 1)
    assert find_best_edge({"A": 5, "B": 1, "C": 3}, "max") == ("A", 5)


def test_find_best_edge_with_filter():

    assert find_best_edge(
        {"A": 5, "B": 1, "C": 3},
        "min",
        {"A": EdgeLabel.BASIC, "B": EdgeLabel.BRANCH, "C": EdgeLabel.BASIC},
        filter_label=EdgeLabel.BASIC,
    ) == ("C", 3)

    assert find_best_edge(
        {"A": 5, "B": 1, "C": 3},
        "max",
        {"A": EdgeLabel.BRANCH, "B": EdgeLabel.BASIC, "C": EdgeLabel.BASIC},
        filter_label=EdgeLabel.BASIC,
    ) == ("C", 3)


def test_find_best_edge_with_filter_empty():

    with pytest.raises(ValueError) as exceptinfo:
        find_best_edge(
            {"A": 5, "B": 1, "C": 3},
            "min",
            {"A": EdgeLabel.BRANCH, "B": EdgeLabel.BRANCH, "C": EdgeLabel.BRANCH},
            filter_label=EdgeLabel.BASIC,
        )
    assert "empty" in str(exceptinfo.value)

    with pytest.raises(ValueError) as exceptinfo:
        find_best_edge(
            {"A": 5, "B": 1, "C": 3},
            "max",
            {"A": EdgeLabel.BRANCH, "B": EdgeLabel.BRANCH, "C": EdgeLabel.BRANCH},
            filter_label=EdgeLabel.BASIC,
        )
    assert "empty" in str(exceptinfo.value)


def test_single_node_wakeup():
    """
    Test with a graph made of a single node.
    Of course, the MST is the single node as well, and we cannot check that edges are
    correctly labelled, as there are no edges !
    The node is requested to wakes up at startup,
    we simply want to make sure the computation terminates.
    """
    c1 = SpanningTreeComputation("c1", [], wakeup_at_start=True)
    a1 = Agent("a1", InProcessCommunicationLayer())
    a1.add_computation(c1)

    run_agents({"a1": a1})
    labels = extract_tree({"c1": c1})
    assert not labels


def test_single_node_auto_wakeup():
    """
    Test with a graph made of a single node, as `test_single_node_wakeup`.
    In this case, the node does not wake up at startup but MUST wake up
    spontaneously after a while.
    """
    c1 = SpanningTreeComputation("c1", [], wakeup_at_start=False)
    a1 = Agent("a1", InProcessCommunicationLayer())
    a1.add_computation(c1)

    run_agents({"a1": a1})
    labels = extract_tree({"c1": c1})
    assert not labels


def test_two_nodes():
    """
    Very simple test: the graph is only made of two nodes and a simple edge,
    thus the spanning graph is the whole edge.
    """
    graph, computations, agents = build_computation([("c1", "c2", 1)], "min")

    initial_wake_up = random.choice(list(computations.values()))
    initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.BRANCH


def test_3_nodes_chain():
    """
    Three nodes, only two edges, the spanning tree is the whole graph.
    * weights are distinct
    * minimum weight spanning tree
    * single spontaneous waking up node
    """
    graph, computations, agents = build_computation(
        [("c1", "c2", 1), ("c2", "c3", 2)], "min"
    )

    initial_wake_up = random.choice(list(computations.values()))
    initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.BRANCH
    assert labels[("c2", "c3")] == EdgeLabel.BRANCH


def test_3_nodes_loop_min():
    """
    Three nodes, three edges
    * minimum weight spanning tree
    * the spanning tree must exclude the most expensive edge
    * single spontaneous waking up node

    """

    edges = [("c1", "c2", 1), ("c2", "c3", 2), ("c3", "c1", 3)]
    graph, computations, agents = build_computation(edges, "min")

    initial_wake_up = random.choice(list(computations.values()))
    initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.BRANCH
    assert labels[("c1", "c3")] == EdgeLabel.REJECTED
    assert labels[("c2", "c3")] == EdgeLabel.BRANCH


def test_3_nodes_loop_min_auto_wakeup():
    """
    Three nodes, three edges
    * minimum weight spanning tree
    * the spanning tree must exclude the most expensive edge
    * single spontaneous waking up node

    """

    edges = [("c1", "c2", 1), ("c2", "c3", 2), ("c3", "c1", 3)]
    graph, computations, agents = build_computation(edges, "min")

    run_agents(agents)
    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.BRANCH
    assert labels[("c1", "c3")] == EdgeLabel.REJECTED
    assert labels[("c2", "c3")] == EdgeLabel.BRANCH


# noinspection DuplicatedCode
def test_3_nodes_as_loop_max():
    """
    Three nodes, three edges
    * minimum weight spanning tree
    * the spanning tree must exclude the least expensive edge
    * single spontaneous waking up node

    """

    edges = [("c1", "c2", 1), ("c2", "c3", 2), ("c3", "c1", 3)]
    graph, computations, agents = build_computation(edges, "max")

    initial_wake_up = random.choice(list(computations.values()))
    initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.REJECTED
    assert labels[("c1", "c3")] == EdgeLabel.BRANCH
    assert labels[("c2", "c3")] == EdgeLabel.BRANCH


def test_3_nodes_as_loop_same_weights():
    """
    Three nodes, three edges
    * minimum weight spanning tree
    *  all edges have the same weights, edge tuple is used to differentiate them
    * single spontaneous waking up node

    """

    edges = [("c1", "c2", 1), ("c2", "c3", 1), ("c3", "c1", 1)]
    graph, computations, agents = build_computation(edges, "min")

    initial_wake_up = random.choice(list(computations.values()))
    initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    assert labels[("c1", "c2")] == EdgeLabel.BRANCH
    assert labels[("c1", "c3")] == EdgeLabel.BRANCH
    assert labels[("c2", "c3")] == EdgeLabel.REJECTED


def test_5_nodes_min(graph_5_nodes):
    """
    5 nodes, 6 edges
    * minimum weight spanning tree (MST)
    * the spanning tree must exclude the two most expensive edges
    * 2 spontaneous waking up node
    """
    edges, excluded = graph_5_nodes
    graph, computations, agents = build_computation(edges,
        "min",
    )

    for initial_wake_up in random.sample(list(computations.values()), 2):
        initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)
    check_mst(labels, excluded)


def test_5_nodes_max(graph_5_nodes):
    """
    5 nodes, 6 edges
    * maximum weight spanning tree (MST)
    * the spanning tree must exclude the two least expensive edges
    * 2 spontaneous waking up node
    """
    edges, excluded = graph_5_nodes
    graph, computations, agents = build_computation(edges,
        "max",
    )

    for initial_wake_up in random.sample(list(computations.values()), 2):
        initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)
    check_mst(labels, [("A", "B"), ("D", "E")])


def test_graph1(graph1_edges):
    """
    Larger graph, still with distinct weights
    Test case from:
    https://github.com/arjun-menon/Distributed-Graph-Algorithms/tree/master/Minimum-Spanning-Tree

    """
    edges, excluded = graph1_edges
    graph, computations, agents = build_computation(edges, "min")

    for initial_wake_up in random.sample(list(computations.values()), 3):
        initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    check_mst(labels, excluded)


def test_graph1_single_wakeup(graph1_edges):
    """
    Larger graph, still with distinct weights
    Test case from:
    https://github.com/arjun-menon/Distributed-Graph-Algorithms/tree/master/Minimum-Spanning-Tree

    """
    edges, excluded = graph1_edges
    graph, computations, agents = build_computation(edges, "min")

    for initial_wake_up in random.sample(list(computations.values()), 1):
        initial_wake_up.wakeup_at_start = True

    run_agents(agents)

    labels = extract_tree(computations)

    check_mst(labels, excluded)


def test_graph1_auto_wakeup(graph1_edges):
    """
    This time, no node is requested to wake up at startup, they should start by
    themself after a while.
    """
    edges, excluded = graph1_edges
    graph, computations, agents = build_computation(edges, "min")

    run_agents(agents)

    labels = extract_tree(computations)

    check_mst(labels, excluded)


def extract_tree(computations):
    """ Extract edge labels from computations """
    # first, check all computation finished properly
    assert all(c.all_labelled for c in computations.values())
    labels = {}
    for c in computations.values():
        for n in c.labels:
            edge = tuple(sorted([c.name, n]))
            if edge in labels:
                # Make sure the label are coherent on both nodes !
                assert (
                    labels[edge] == c.labels[n]
                ), f"incoherent edeg label {edge} {c.name}"
            else:
                labels[edge] = c.labels[n]
    return labels


def check_mst(labels: Dict, expected_rejected):
    """
    Check that the labels decribe the expected MST.

    Parameters
    ----------
    labels: dict
        dict {edge : weight}, obtained using `extract_tree`
    expected_rejected : iterable
        iterable of edges that should be labelled as rejected

    Returns
    -------

    """
    expected_rejected = [tuple(sorted([n1, n2])) for n1, n2 in expected_rejected]
    for edge, label in labels.items():
        if edge in expected_rejected:
            assert label == EdgeLabel.REJECTED, f"{edge} should be rejected"
        else:
            assert label == EdgeLabel.BRANCH, f"{edge} should be accepted"


def build_computation(edges, mode):
    """ Build computations and agents for computing the min spanning tree for a
    weighted graph."""
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)

    computations = {}
    agents = {}
    for node in graph.nodes:
        node_edges = [
            (neighbor, graph[node][neighbor]["weight"]) for neighbor in graph[node]
        ]

        computations[node] = SpanningTreeComputation(node, node_edges, mode=mode)
        a = Agent(f"a_{node}", InProcessCommunicationLayer())
        a.add_computation(computations[node])
        agents[node] = a

    for node in graph.nodes:
        for neighbor in graph[node]:
            agents[node].discovery.register_computation(
                neighbor, agents[neighbor].name, agents[neighbor].address
            )
    return graph, computations, agents


def run_agents(agents):
    for a in agents.values():
        a.start(run_computations=True)

    active_agents = list(agents.values())
    while active_agents:
        for a in active_agents:
            if not a.has_running_computations():
                a.stop()
                active_agents.remove(a)
        sleep(0.02)

    for a in agents.values():
        a.join()


@pytest.fixture
def graph_5_nodes():
    edges = [
        ("A", "B", 2),
        ("A", "C", 6),
        ("B", "C", 4),
        ("B", "D", 3),
        ("C", "E", 5),
        ("D", "E", 1),
    ]
    excluded = [("A", "C"), ("C", "E")]
    return edges, excluded


@pytest.fixture
def graph1_edges():
    edges = [
        ("A", "F", 2),
        ("F", "G", 7),
        ("G", "H", 15),
        ("H", "J", 13),
        ("J", "I", 9),
        ("I", "C", 18),
        ("C", "B", 17),
        ("B", "A", 3),
        ("E", "F", 1),
        ("E", "G", 6),
        ("E", "H", 5),
        ("E", "I", 10),
        ("E", "D", 11),
        ("I", "H", 12),
        ("D", "I", 4),
        ("D", "C", 8),
        ("D", "B", 16),
    ]
    excluded = [
        ("C", "I"),
        ("C", "B"),
        ("B", "D"),
        ("D", "E"),
        ("H", "J"),
        ("H", "I"),
        ("H", "G"),
        ("F", "G"),
    ]
    return edges, excluded
