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


import unittest
from collections import namedtuple

from pydcop.utils.graphs import as_bipartite_graph, Node, \
    find_furthest_node, \
    calc_diameter, as_networkx_graph, all_pairs, cycles_count, graph_diameter
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import UnaryFunctionRelation, \
    NAryFunctionRelation


class AsGraphTests(unittest.TestCase):

    def test_1var_1rel(self):
        domain = list(range(10))
        l1 = Variable('l1', domain)
        rel_l1 = UnaryFunctionRelation('rel_l1', l1, lambda x: x)

        nodes = as_bipartite_graph([l1], [rel_l1])

        self.assertEqual(len(nodes), 2)
        var_nodes = [n for n in nodes if n.type == 'VARIABLE']
        rel_nodes = [n for n in nodes if n.type == 'CONSTRAINT']

        self.assertEqual(len(var_nodes), 1)
        self.assertEqual(len(rel_nodes), 1)

    def test_3var_1rel(self):
        domain = list(range(10))
        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        rel = NAryFunctionRelation(lambda x, y, z: 0, [l1, l2, l3], name='rel')

        nodes = as_bipartite_graph([l1, l2, l3], [rel])

        self.assertEqual(len(nodes), 4)
        var_nodes = [n for n in nodes if n.type == 'VARIABLE']
        rel_nodes = [n for n in nodes if n.type == 'CONSTRAINT']

        self.assertEqual(len(var_nodes), 3)
        self.assertEqual(len(rel_nodes), 1)

        self.assertEqual(len(rel_nodes[0].neighbors), 3)
        self.assertIn(var_nodes[0], rel_nodes[0].neighbors)
        self.assertIn(var_nodes[1], rel_nodes[0].neighbors)
        self.assertIn(var_nodes[2], rel_nodes[0].neighbors)


class FindFurthestInLoopyGrap(unittest.TestCase):

    def setUp(self):
        Content = namedtuple('Content', ['name'])
        self.n1 = Node(Content('n1'))
        self.n2 = Node(Content('n2'))
        self.n3 = Node(Content('n3'))
        self.n4 = Node(Content('n4'))
        self.n5 = Node(Content('n5'))

        self.n1.add_neighbors(self.n2)
        self.n1.add_neighbors(self.n3)
        self.n2.add_neighbors(self.n4)
        self.n3.add_neighbors(self.n4)
        self.n3.add_neighbors(self.n5)

        self.nodes = [self.n1, self.n2, self.n3, self.n4, self.n5]

    def test_from_n1(self):
        furthest, distance = find_furthest_node(self.n1, self.nodes)
        self.assertTrue(furthest == self.n4 or furthest == self.n5)
        self.assertEqual(distance, 2)

    def test_from_n2(self):
        furthest, distance = find_furthest_node(self.n2, self.nodes)
        self.assertTrue(furthest == self.n5)
        self.assertEqual(distance, 3)

    def test_from_n3(self):
        furthest, distance = find_furthest_node(self.n3, self.nodes)
        self.assertTrue(furthest == self.n2)
        self.assertEqual(distance, 2)

    def test_from_n4(self):
        furthest, distance = find_furthest_node(self.n4, self.nodes)
        self.assertTrue(furthest == self.n5 or self.n1 == furthest)
        self.assertEqual(distance, 2)

    def test_from_n5(self):
        furthest, distance = find_furthest_node(self.n5, self.nodes)
        self.assertTrue(furthest == self.n2)
        self.assertEqual(distance, 3)


class GraphDiameterTests(unittest.TestCase):

    def test_furthest_node(self):
        Content = namedtuple('Content', ['name'])
        n1 = Node(Content('n1'))
        n2 = Node(Content('n2'))
        n3 = Node(Content('n3'))
        n4 = Node(Content('n4'))

        n1.add_neighbors(n2)
        n2.add_neighbors(n3)
        n2.add_neighbors(n4)

        _, d = find_furthest_node(n1, [n1, n2, n3, n4])
        self.assertEqual(d, 2)

        _, d = find_furthest_node(n2, [n1, n2, n3, n4])
        self.assertEqual(d, 1)

    def test_diameter(self):
        Content = namedtuple('Content', ['name'])
        n1 = Node(Content('n1'))
        n2 = Node(Content('n2'))
        n3 = Node(Content('n3'))
        n4 = Node(Content('n4'))

        n1.add_neighbors(n2)
        n2.add_neighbors(n3)
        n2.add_neighbors(n4)

        nodes = [n1, n2, n3, n4]

        self.assertEqual(calc_diameter(nodes), 2)

    def test_diameter_5nodes(self):
        Content = namedtuple('Content', ['name'])
        n1 = Node(Content('n1'))
        n2 = Node(Content('n2'))
        n3 = Node(Content('n3'))
        n4 = Node(Content('n4'))
        n5 = Node(Content('n5'))
        n6 = Node(Content('n6'))

        n1.add_neighbors(n2)
        n2.add_neighbors(n3)
        n2.add_neighbors(n4)
        n4.add_neighbors(n5)
        n5.add_neighbors(n6)

        nodes = [n1, n2, n3, n4, n5, n6]

        self.assertEqual(calc_diameter(nodes), 4)

    def test_diameter_loop(self):
        Content = namedtuple('Content', ['name'])
        n1 = Node(Content('n1'))
        n2 = Node(Content('n2'))
        n3 = Node(Content('n3'))
        n4 = Node(Content('n4'))
        n5 = Node(Content('n5'))

        n1.add_neighbors(n2)
        n1.add_neighbors(n3)
        n2.add_neighbors(n4)
        n3.add_neighbors(n4)
        n3.add_neighbors(n5)

        nodes = [n1, n2, n3, n4, n5]

        # FIXME: cycles count only works on trees !!
        # self.assertEqual(calc_diameter(nodes), 3)


class NetworkXTests(unittest.TestCase):

    def test_pairs_2elt(self):
        elts = ['a', 'b']
        pairs = all_pairs(elts)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ('a', 'b'))

    def test_pairs_3elt(self):
        elts = ['a', 'b', 'c']
        pairs = all_pairs(elts)
        self.assertEqual(len(pairs), 3)
        self.assertIn(('a', 'b'), pairs)
        self.assertIn(('a', 'c'), pairs)
        self.assertIn(('b', 'c'), pairs)

    def test_pairs_delt(self):
        elts = ['a', 'b', 'c', 'd']
        pairs = all_pairs(elts)
        self.assertEqual(len(pairs), 6)
        self.assertIn(('a', 'b'), pairs)
        self.assertIn(('a', 'c'), pairs)
        self.assertIn(('a', 'd'), pairs)
        self.assertIn(('b', 'c'), pairs)
        self.assertIn(('b', 'd'), pairs)
        self.assertIn(('c', 'd'), pairs)

    def test_convert_graph_simple(self):
        domain = list(range(10))
        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: 0, [l2, l3], name='r2')
        r3 = NAryFunctionRelation(lambda x, y: 0, [l1, l3], name='r3')

        graph = as_networkx_graph([l1, l2, l3], [r1, r2, r3])

        print(graph.edges())
        print(graph.nodes())

        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 3)

    def test_convert_graph(self):
        domain = list(range(10))
        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        l4 = Variable('l4', domain)

        # 4-ary relation : iot defines a clique with l1, l2, l3, l4
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2, l3, l4], name='r1')

        graph = as_networkx_graph([l1, l2, l3], [r1])

        print(graph.edges())
        print(graph.nodes())

        self.assertEqual(len(graph.nodes()), 4)
        self.assertEqual(len(graph.edges()), 6)

    def test_count_cycle_none(self):
        domain = list(range(10))

        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: 0, [l2, l3], name='r2')

        n = cycles_count([l1, l2, l3], [r1, r2])

        self.assertEqual(n, 0)

    def test_count_cycle_one(self):
        domain = list(range(10))

        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: 0, [l2, l3], name='r2')
        r3 = NAryFunctionRelation(lambda x, y: 0, [l1, l3], name='r3')

        n = cycles_count([l1, l2, l3], [r1, r2, r3])

        self.assertEqual(n, 1)

    def test_count_cycle_clique(self):
        domain = list(range(10))

        l1 = Variable('l1', domain)
        l2 = Variable('l2', domain)
        l3 = Variable('l3', domain)
        l4 = Variable('l4', domain)
        r1 = NAryFunctionRelation(lambda x, y, z, w: 0, [l1, l2, l3, l4],
                                  name='r1')

        n = cycles_count([l1, l2, l3, l4], [r1])

        self.assertEqual(n, 3)

    def test_diameter_simple(self):
        l1 = Variable('l1', [])
        l2 = Variable('l2', [])
        l3 = Variable('l3', [])
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: 0, [l2, l3], name='r2')

        d = graph_diameter([l1, l2, l3], [r1, r2])
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0], 2)

    def test_diameter_simple2(self):
        l1 = Variable('l1', [])
        l2 = Variable('l2', [])
        l3 = Variable('l3', [])
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: 0, [l2, l3], name='r2')
        r3 = NAryFunctionRelation(lambda x, y: 0, [l1, l3], name='r3')

        d = graph_diameter([l1, l2, l3], [r1, r2, r3])
        self.assertListEqual(d, [1])

    def test_diameter_simple3(self):
        l1 = Variable('l1', [])
        l2 = Variable('l2', [])
        l3 = Variable('l3', [])
        r1 = NAryFunctionRelation(lambda x, y: 0, [l1, l2], name='r1')

        g = as_networkx_graph([l1, l2, l3], [r1])

        d = graph_diameter([l1, l2, l3], [r1])
        self.assertListEqual(sorted(d), [0, 1])
