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

from pydcop.computations_graph.pseudotree import _find_neighbors_relations, \
    _BuildingNode, \
    _generate_dfs_tree, _visit_tree, build_computation_graph, \
    _filter_relation_to_lowest_node, PseudoTreeNode, PseudoTreeLink
from pydcop.dcop.objects import Variable, VariableDomain
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.relations import NAryFunctionRelation, relation_from_str
from pydcop.utils.simple_repr import simple_repr, from_repr


class DfsTreeGenerationTests(unittest.TestCase):

    def test_find_neighbors_relations(self):
        domain = ['a', 'b', 'c']

        n1 = _BuildingNode(Variable('x1', domain))
        n2 = _BuildingNode(Variable('x2', domain))
        n3 = _BuildingNode(Variable('x3', domain))
        nodes = [n1, n2, n3]

        r1 = NAryFunctionRelation(lambda x, y: x + y,
                                  [n1.variable, n2.variable],
                                  name='r1')
        relations = [r1]

        self.assertEqual(_find_neighbors_relations(n1, relations, nodes),
                         ([n2], [r1]))
        self.assertEqual(_find_neighbors_relations(n2, relations, nodes),
                         ([n1], [r1]))
        self.assertEqual(_find_neighbors_relations(n3, relations, nodes),
                         ([], []))

    def test_root(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: x - y, [x2, x3], name='r2')
        relations = [r1, r2]

        tree_root = _generate_dfs_tree(variables, relations, x1)
        self.assertEqual(tree_root.variable, x1)

        tree_root = _generate_dfs_tree(variables, relations)
        self.assertIn(tree_root.variable, variables)

    def test_2nodes_tree(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        variables = [x1, x2]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        relations = [r1]

        root = _generate_dfs_tree(variables, relations, root=x1)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 1)
        node = root.children[0]
        self.assertEqual(node.variable, x2)
        self.assertEqual(root.pseudo_children, [])
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(node.parent, root)
        self.assertEqual(node.children, [])
        self.assertEqual(node.pseudo_children, [])
        self.assertEqual(node.pseudo_parents, [])
        self.assertEqual(node.relations, [r1])

        check_tree(root)

    def test_3nodes_tree(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: x + y, [x1, x3], name='r1')
        relations = [r1, r2]

        root = _generate_dfs_tree(variables, relations, root=x1)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 2)
        c1 = root.children[0]
        c2 = root.children[1]
        self.assertIn(c1.variable, [x2, x3])
        self.assertIn(c2.variable, [x2, x3])
        self.assertNotEqual(c1, c2)
        self.assertEqual(root.pseudo_children, [])
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(c1.parent, root)
        self.assertEqual(c1.children, [])
        self.assertEqual(c1.pseudo_children, [])
        self.assertEqual(c1.pseudo_parents, [])

        self.assertEqual(c2.parent, root)
        self.assertEqual(c2.children, [])
        self.assertEqual(c2.pseudo_children, [])
        self.assertEqual(c2.pseudo_parents, [])

        check_tree(root)

    def test_3nodes_tree_cycle(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: x + y, [x1, x3], name='r2')
        r3 = NAryFunctionRelation(lambda x, y: x + y, [x2, x3], name='r3')
        relations = [r1, r2, r3]

        root = _generate_dfs_tree(variables, relations, root=x1)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        c1 = root.children[0]
        self.assertIn(c1.variable, [x2, x3])
        self.assertEqual(len(root.pseudo_children), 1)
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(c1.parent, root)
        self.assertEqual(len(c1.children), 1)
        c2 = c1.children[0]
        self.assertEqual(c2.children, [])
        self.assertEqual(c2.pseudo_children, [])
        self.assertEqual(c2.pseudo_parents, [root])

        check_tree(root)

    def test_3nodes_tree_cycle_3ary(self):
        # A graph with 3 variables and a single 3-ary relation.

        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]
        r1 = NAryFunctionRelation(lambda x, y, z: x + y + z, [x1, x2, x3],
                                  name='r1')
        relations = [r1]

        root = _generate_dfs_tree(variables, relations, root=x1)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 1)
        c1 = root.children[0]
        self.assertIn(c1.variable, [x2, x3])
        self.assertEqual(len(root.pseudo_children), 1)
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(c1.parent, root)
        self.assertEqual(len(c1.children), 1)
        c2 = c1.children[0]
        self.assertEqual(c2.children, [])
        self.assertEqual(c2.pseudo_children, [])
        self.assertEqual(c2.pseudo_parents, [root])

        self.assertEqual(len(c1.relations), 1)
        self.assertEqual(len(c2.relations), 1)

        check_tree(root)

    def test_4nodes(self):
        # Graph with 4 nodes, one cycle
        #
        #       x1---X3
        #        \  /
        #         x2---x4

        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        x4 = Variable('x4', domain)
        variables = [x1, x2, x3, x4]

        binary_func = lambda x, y: x + y
        r1 = NAryFunctionRelation(binary_func, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(binary_func, [x1, x3], name='r2')
        r3 = NAryFunctionRelation(binary_func, [x2, x3], name='r3')
        r4 = NAryFunctionRelation(binary_func, [x2, x4], name='r4')
        relations = [r1, r2, r3, r4]

        root = _generate_dfs_tree(variables, relations, root=x1)

        check_tree(root)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 2)

    def test_visit_tree(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        x4 = Variable('x4', domain)
        variables = [x1, x2, x3, x4]

        binary_func = lambda x, y: x + y
        r1 = NAryFunctionRelation(binary_func, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(binary_func, [x1, x3], name='r2')
        r3 = NAryFunctionRelation(binary_func, [x2, x3], name='r3')
        r4 = NAryFunctionRelation(binary_func, [x2, x4], name='r4')
        relations = [r1, r2, r3, r4]

        root = _generate_dfs_tree(variables, relations, root=x1)

        for node in _visit_tree(root):
            self.assertIn(node.variable, variables)
            variables.remove(node.variable)

        self.assertEqual(len(variables), 0)


def check_tree(tree_root, visited=None):
    """
    Perform some basic structural check on a DFS tree.

    :param tree_root:
    :param visited:
    :return:
    """
    if visited is None:
        visited = []
    for pp in tree_root.pseudo_parents:
        if tree_root not in pp.pseudo_children:
            return False
        if pp not in visited:
            return False

    for pc in tree_root.pseudo_children:
        if tree_root not in pc.pseudo_parents:
            return False

    for c in tree_root.children:
        if c in visited:
            return False
        visited.append(c)
        check_tree(c, visited)
    return True


class PseudoTreeGeneration(unittest.TestCase):

    def test_build_single_var(self):

        v1 = Variable('v1', [1,2,3])
        dcop = DCOP('test', 'min')
        dcop.variables = {'v1': v1}

        cg = build_computation_graph(dcop)

        self.assertEqual(len(cg.nodes), 1)
        self.assertEqual(cg.computation('v1').variable, v1)
        self.assertEqual(len(cg.links), 0)

    def test_build_two_var(self):

        v1 = Variable('v1', [1,2,3])
        v2 = Variable('v2', [1,2,3])
        c1 = relation_from_str('c1', 'v1 + v2 ', [v1, v2])
        dcop = DCOP('test', 'min')
        dcop.add_constraint(c1)

        cg = build_computation_graph(dcop)

        self.assertEqual(len(cg.nodes), 2)
        self.assertEqual(cg.computation('v1').variable, v1)
        self.assertEqual(cg.computation('v2').variable, v2)

        # one parent link, one children link
        self.assertEqual(len(cg.links), 2)

    def test_3nodes_tree_cycle(self):
        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(lambda x, y: x + y, [x1, x3], name='r2')
        r3 = NAryFunctionRelation(lambda x, y: x + y, [x2, x3], name='r3')
        relations = [r1, r2, r3]

        dcop = DCOP('test', 'min')
        dcop.add_constraint(r1)
        dcop.add_constraint(r2)
        dcop.add_constraint(r3)

        cg = build_computation_graph(dcop)
        self.assertEqual(len(cg.nodes), len(variables))

        self.assertEqual(len(cg.roots), 1)
        # All variables have the same number of neighbors, they could all be
        #  root
        self.assertIn(cg.roots[0].variable, [x1, x2, x3])
        self.assertEqual(cg.roots[0].parent, None)

        root = _generate_dfs_tree(variables, relations, root=x1)
        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)


class DfsTreeDpopTests(unittest.TestCase):

    def test_2nodes_tree_relation_at_bottom(self):

        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        variables = [x1, x2]

        r1 = NAryFunctionRelation(lambda x, y: x + y, [x1, x2], name='r1')
        relations = [r1]

        root = _generate_dfs_tree(variables, relations, root=x1)
        _filter_relation_to_lowest_node(root)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 0)
        node = root.children[0]
        self.assertEqual(node.variable, x2)
        self.assertEqual(root.pseudo_children, [])
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(node.parent, root)
        self.assertEqual(node.children, [])
        self.assertEqual(node.pseudo_children, [])
        self.assertEqual(node.pseudo_parents, [])
        self.assertEqual(node.relations, [r1])

        check_tree(root)

    def test_3nodes_tree_cycle_3ary_rel_bottom(self):
        # A graph with 3 variables and a single 3-ary relation.

        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        variables = [x1, x2, x3]
        r1 = NAryFunctionRelation(lambda x, y, z: x + y + z, [x1, x2, x3],
                                  name='r1')
        relations = [r1]

        root = _generate_dfs_tree(variables, relations, root=x1)
        _filter_relation_to_lowest_node(root)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 0)
        c1 = root.children[0]
        self.assertIn(c1.variable, [x2, x3])
        self.assertEqual(len(root.pseudo_children), 1)
        self.assertEqual(root.pseudo_parents, [])

        self.assertEqual(c1.parent, root)
        self.assertEqual(len(c1.children), 1)
        c2 = c1.children[0]
        self.assertEqual(c2.children, [])
        self.assertEqual(c2.pseudo_children, [])
        self.assertEqual(c2.pseudo_parents, [root])

        self.assertEqual(len(c1.relations) + len(c2.relations), 1)

        check_tree(root)

    def test_4nodes(self):
        # Graph with 4 nodes, one cycle
        #
        #       x1---X3
        #        \  /
        #         x2---x4

        domain = ['a', 'b', 'c']
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)
        x4 = Variable('x4', domain)
        variables = [x1, x2, x3, x4]

        def binary_func(x, y):
            return x + y

        r1 = NAryFunctionRelation(binary_func, [x1, x2], name='r1')
        r2 = NAryFunctionRelation(binary_func, [x1, x3], name='r2')
        r3 = NAryFunctionRelation(binary_func, [x2, x3], name='r3')
        r4 = NAryFunctionRelation(binary_func, [x2, x4], name='r4')
        relations = [r1, r2, r3, r4]

        root = _generate_dfs_tree(variables, relations, root=x1)
        _filter_relation_to_lowest_node(root)

        check_tree(root)

        self.assertEqual(root.variable, x1)
        self.assertEqual(root.parent, None)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(len(root.relations), 0)


class TestPseudoTreeSimpleRepr(unittest.TestCase):

    def test_node(self):
        v1 = Variable('v1', [1, 2, 3])
        node = _BuildingNode(v1)

    def test_link_simple_repr(self):

        l1 = PseudoTreeLink('parent', 'v1', 'v2')

        r = simple_repr(l1)

    def test_link_from_repr(self):
        l1 = PseudoTreeLink('parent', 'v1', 'v2')

        r = simple_repr(l1)
        l2 = from_repr(r)
        self.assertEqual(l1, l2)


class TestMetrics(unittest.TestCase):

    def test_density_two_var_one_factor(self):
        dcop = DCOP('test', 'min')
        d1 = VariableDomain('d1', '--', [1, 2, 3])
        v1 = Variable('v1', d1)
        v2 = Variable('v2', d1)
        c1 = relation_from_str('c1', '0.5 * v1 + v2', [v1, v2])

        dcop.add_constraint(c1)

        g = build_computation_graph(dcop)

        self.assertEqual(g.density(), 2/2)