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
from unittest.mock import MagicMock

import numpy as np
import pytest

from pydcop.algorithms import dpop
from pydcop.algorithms.dpop import DpopMessage
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import NAryMatrixRelation, AsNAryFunctionRelation


@pytest.mark.skip
def test_communicatino_load():
    dpop.communication_load()

@pytest.mark.skip
def test_computation_memory():
    dpop.computation_memory()

class JoinRelationsTestCase(unittest.TestCase):

    def test_arity_bothsamevar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1])
        u2 = NAryMatrixRelation([x1])

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 1)

    def test_arity_2diffvar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1])

        x2 = Variable('x2', ['1', '2'])
        u2 = NAryMatrixRelation([x2])

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 2)

    def test_arity_3diffvar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        x2 = Variable('x2', ['1', '2'])
        u1 = NAryMatrixRelation([x1, x2])

        x3 = Variable('x3', ['z', 'y'])
        u2 = NAryMatrixRelation([x2, x3])

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 3)

    def test_join_bothsamevar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))
        u2 = NAryMatrixRelation([x1], np.array([1, 2, 3], np.int8))

        # x1 = Variable('x1', ['a', 'b', 'c'])
        # u1 = dpop.NAryRelation([x1], np.array([1, 2, 3], np.int8))

        self.assertEqual(u1.get_value_for_assignment(['b']), 2)

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 1)
        self.assertEqual(u_j.get_value_for_assignment(['a']), 2)
        self.assertEqual(u_j.get_value_for_assignment(['b']), 4)
        self.assertEqual(u_j.get_value_for_assignment(['c']), 6)

    def test_join_2diffvar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        x2 = Variable('x2', ['1', '2'])
        u2 = NAryMatrixRelation([x2], np.array([1, 3], np.int8))

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 2)

        self.assertEqual(u_j.get_value_for_assignment(['a', '1']), 3)
        self.assertEqual(u_j.get_value_for_assignment(['c', '2']), 11)
        self.assertEqual(u_j.get_value_for_assignment(['b', '1']), 5)

    def test_join_3diffvar(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        x2 = Variable('x2', ['1', '2'])
        u1 = NAryMatrixRelation([x1, x2], np.array([[2, 16],
                                                    [4, 32],
                                                    [8, 64]], np.int8))

        x3 = Variable('x3', ['z', 'y'])
        u2 = NAryMatrixRelation([x2, x3], np.array([[1, 5], [3, 7]], np.int8))

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 3)
        self.assertEqual(u_j.dimensions, [x1, x2, x3])

        self.assertEqual(u_j.get_value_for_assignment(['a', '1', 'z']), 3)
        self.assertEqual(u_j.get_value_for_assignment(['b', '2', 'y']), 39)

    def test_join_with_no_var_rel(self):
        # join a relation with a relation with no dimension

        x1 = Variable('x1', ['a', 'b', 'c'])
        x2 = Variable('x2', ['1', '2'])
        u1 = NAryMatrixRelation([x1, x2], np.array([[2, 16],
                                                    [4, 32],
                                                    [8, 64]], np.int8))
        u2 = NAryMatrixRelation([])

        u_j = dpop.join_utils(u1, u2)

        self.assertEqual(u_j.arity, 2)
        self.assertEqual(u_j.dimensions, [x1, x2])

        self.assertEqual(u_j.get_value_for_assignment(['a', '1']), 2)
        self.assertEqual(u_j.get_value_for_assignment(['b', '2']), 32)

    def test_join_different_order(self):
        # Test joining 2 relations that do not declare their variable in the
        # same order

        x1 = Variable('x1', [0, 1, 2])
        x2 = Variable('x2', [0, 1, 2])

        @AsNAryFunctionRelation(x1, x2)
        def u1(x, y):
            return x+y

        @AsNAryFunctionRelation(x2, x1)
        def u2(x, y):
            return x - y

        j = dpop.join_utils(u1, u2)

        self.assertEqual(j(1, 1), 2)
        self.assertEqual(j(1, 2), 4)


class ProjectionTestCase(unittest.TestCase):

    def test_projection_oneVarRel(self):

        # u1 is a relation with a single variable :
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        # take the projection of u1 along x1
        p = dpop.projection(u1, x1)

        # the dimension must be one less than the dimension of u1
        self.assertEqual(p.arity, 0)

        # this means that p is actually a signle value, corresponding to the
        # max of u1
        self.assertEqual(p.get_value_for_assignment(), 8)

    def test_projection_min_oneVarRel(self):
        # u1 is a relation with a single variable :
        x1 = Variable('x1', ['a', 'b', 'c'])
        u1 = NAryMatrixRelation([x1], np.array([2, 4, 8], np.int8))

        # take the projection of u1 along x1
        p = dpop.projection(u1, x1, mode='min')

        # the dimension must be one less than the dimension of u1
        self.assertEqual(p.arity, 0)

        # this means that p is actually a signle value, corresponding to the
        # max of u1
        self.assertEqual(p.get_value_for_assignment(), 2)

    def test_projection_twoVarsRel(self):

        x1 = Variable('x1', ['a', 'b', 'c'])
        x2 = Variable('x2', ['1', '2'])
        u1 = NAryMatrixRelation([x1, x2], np.array([[2, 16],
                                                    [4, 32],
                                                    [8, 64]], np.int8))

        # take the projection of u1 along x1
        p = dpop.projection(u1, x1)

        # the dimension must be one less than the dimension of u1, it should
        # contain only x2
        self.assertEqual(p.arity, 1)
        self.assertListEqual(p.dimensions, [x2])

        # the max of u1 when setting x2<-1 is 8
        self.assertEqual(p.get_value_for_assignment(['1']), 8)

        # the max of u1 when setting x2<-2 is 64
        self.assertEqual(p.get_value_for_assignment(['2']), 64)

    def test_projection_min_twoVarsRel(self):
        x1 = Variable('x1', ['a', 'b', 'c'])
        x2 = Variable('x2', ['1', '2'])
        u1 = NAryMatrixRelation([x1, x2], np.array([[2, 16],
                                                    [4, 32],
                                                    [8, 64]], np.int8))

        # take the projection of u1 along x1
        p = dpop.projection(u1, x1, mode='min')

        # the dimension must be one less than the dimension of u1, it should
        # contain only x2
        self.assertEqual(p.arity, 1)
        self.assertListEqual(p.dimensions, [x2])

        # the min of u1 when setting x2<-1 is 2
        self.assertEqual(p.get_value_for_assignment(['1']), 2)

        # the min of u1 when setting x2<-2 is 16
        self.assertEqual(p.get_value_for_assignment(['2']), 16)


class AddVarToAssignmentTestCase(unittest.TestCase):

    def test_add_var_to_assignment_oneVar(self):
        x1 = Variable('x1', ['a1', 'a2', 'a3'])

        # Add the var x1 with value 'a3' to an empty assignment
        assignt = []
        assignt = dpop._add_var_to_assignment(assignt, [x1], x1, 'a3')

        self.assertEqual(len(assignt), 1)

    def test_add_var_to_assignment_twoVar(self):
        x1 = Variable('x1', ['a1', 'a2', 'a3'])
        x2 = Variable('x2', ['b1', 'b2'])

        # Add the var x1 with value 'a3' to an empty assignment
        assignt = ['b2']
        assignt = dpop._add_var_to_assignment(assignt, [x1, x2], x1, 'a3')

        self.assertEqual(len(assignt), 2)
        self.assertListEqual(assignt, ['a3', 'b2'])

    def test_add_var_to_assignment_ThreeVars_middle(self):
        x1 = Variable('x1', ['a1', 'a2', 'a3'])
        x2 = Variable('x2', ['b1', 'b2'])
        x3 = Variable('x3', ['c1', 'c2'])

        assignt = ['a1', 'c2']
        assignt = dpop._add_var_to_assignment(assignt, [x1, x2, x3], x2, 'b2')

        self.assertEqual(len(assignt), 3)
        self.assertListEqual(assignt, ['a1', 'b2', 'c2'])

    def test_add_var_to_assignment_ThreeVars_start(self):
        x1 = Variable('x1', ['a1', 'a2', 'a3'])
        x2 = Variable('x2', ['b1', 'b2'])
        x3 = Variable('x3', ['c1', 'c2'])

        assignt = ['b1', 'c2']
        assignt = dpop._add_var_to_assignment(assignt, [x1, x2, x3], x1, 'a2')

        self.assertEqual(len(assignt), 3)
        self.assertListEqual(assignt, ['a2', 'b1', 'c2'])

    def test_add_var_to_assignment_ThreeVars_end(self):
        x1 = Variable('x1', ['a1', 'a2', 'a3'])
        x2 = Variable('x2', ['b1', 'b2'])
        x3 = Variable('x3', ['c1', 'c2'])

        assignt = ['a1', 'b2']
        assignt = dpop._add_var_to_assignment(assignt, [x1, x2, x3], x3, 'c2')

        self.assertEqual(len(assignt), 3)
        self.assertListEqual(assignt, ['a1', 'b2', 'c2'])


class DummySender(object):

    def __init__(self):
        self.util_sender_var = None
        self.util_dest_var = None
        self.util_msg_data = None
        self.value_sender_var = None
        self.value_dest_var = None
        self.value_msg_data = None

    def __call__(self, sender_var, dest_var, msg, prio=None, on_error=None):
        if msg.type == 'UTIL':
            self.util_sender_var = sender_var
            self.util_dest_var = dest_var
            self.util_msg_data = msg.content
        elif msg.type == 'VALUE':
            self.value_sender_var = sender_var
            self.value_dest_var = dest_var
            self.value_msg_data = msg.content


class AlgoExampleTwoVarsTestcase(unittest.TestCase):
    """
    Test case with a very simplistic setup with only two vars and one relation
     a0 -> a1

    """

    def setUp(self):
        self.x0 = Variable('x0', ['a', 'b'])
        self.x1 = Variable('x1', ['a', 'b'])

        self.r0_1 = NAryMatrixRelation([self.x0, self.x1], np.array([[1, 2],
                                                                     [4, 3]]))

        self.sender0 = DummySender()
        self.sender1 = DummySender()
        compdef = MagicMock()
        compdef.algo.algo = 'dpop'
        compdef.algo.mode = 'max'
        self.a0 = dpop.DpopAlgo(self.x0,
                                parent=None, children=[self.x1.name],
                                constraints=[],
                                msg_sender=self.sender0, comp_def=compdef)
        self.a1 = dpop.DpopAlgo(self.x1, parent=self.x0.name, children=[],
                                constraints=[self.r0_1],
                                msg_sender=self.sender1, comp_def=compdef)

    def test_onstart_two_vars(self):

        # a0 is the root, must not send any message on start
        self.a0.on_start()
        self.assertEqual(self.sender0.util_msg_data, None)
        self.assertEqual(self.sender0.value_msg_data, None)

        # a1 is the leaf, sends a util message
        self.a1.on_start()
        print(self.sender1.util_msg_data)

        self.assertEqual(self.sender1.util_msg_data('a'), 2)
        self.assertEqual(self.sender1.util_msg_data('b'), 4)

    def test_on_util_root_two_vars(self):

        # Testing that the root select the correct variable when receiving
        # the util message from its only child.

        self.a0.on_start()
        self.a1.on_start()

        u1_0 = NAryMatrixRelation([self.x0], np.array([2, 4]))
        msg = DpopMessage('UTIL', u1_0)
        self.a0._on_util_message(self.x1.name, msg, 0)

        # a0 id the root, when receiving UTIL message it must compute its own
        #  optimal value and send a value message
        msg = DpopMessage('VALUE', ([self.x0], ['b']))
        self.assertEqual(self.sender0.value_msg_data, msg.content)
        self.assertEqual(self.a0.current_value, 'b')

    def test_value_leaf_two_vars(self):

        self.a0.on_start()
        self.a1.on_start()

        msg = DpopMessage('VALUE', ([self.x0], ['b']))
        self.a1._on_value_message(self.a0, msg, 0)
        self.assertEqual(self.a1.current_value, 'a')


class AlgoExampleThreeVarsTestcase(unittest.TestCase):
    """
    Test case with a very simplistic setup with only two vars and one relation
     a0 -> a1
        -> a2

    """

    def setUp(self):
        self.x0 = Variable('x0', ['a', 'b'])
        self.x1 = Variable('x1', ['a', 'b'])
        self.x2 = Variable('x2', ['a', 'b'])

        self.r0_1 = NAryMatrixRelation(
            [self.x0, self.x1],
            np.array([[1, 2], [2, 3]]))
        self.r0_2 = NAryMatrixRelation(
            [self.x0, self.x2],
            np.array([[5, 2], [3, 1]]))

        self.sender0 = DummySender()
        self.sender1 = DummySender()
        self.sender2 = DummySender()
        compdef = MagicMock()
        compdef.algo.algo = 'dpop'
        compdef.algo.mode = 'max'

        self.a0 = dpop.DpopAlgo(self.x0, parent=None,
                                children=[self.x1.name, self.x2.name],
                                constraints=[],
                                msg_sender=self.sender0,
                                comp_def=compdef)
        self.a1 = dpop.DpopAlgo(self.x1, parent=self.x0.name,
                                children=[], constraints=[self.r0_1],
                                msg_sender=self.sender1,
                                comp_def=compdef)
        self.a2 = dpop.DpopAlgo(self.x2, parent=self.x0.name,
                                children=[], constraints=[self.r0_2],
                                msg_sender=self.sender2,
                                comp_def=compdef)

    def test_on_start(self):
        # a0 is the root, must not send any message on start
        self.a0.on_start()
        self.assertEqual(self.sender0.util_msg_data, None)
        self.assertEqual(self.sender0.value_msg_data, None)

        # a1 is a leaf, sends a util message
        self.a1.on_start()
        print(self.sender1.util_msg_data)

        self.assertEqual(self.sender1.util_msg_data('a'), 2)
        self.assertEqual(self.sender1.util_msg_data('b'), 3)

        self.a2.on_start()
        print(self.sender2.util_msg_data)

        self.assertEqual(self.sender2.util_msg_data('a'), 5)
        self.assertEqual(self.sender2.util_msg_data('b'), 3)

    def test_on_util_root_two_vars(self):
        # Testing that the root select the correct variable when receiving
        # the util message from its only child.

        self.a0.on_start()
        self.a1.on_start()

        u1_0 = NAryMatrixRelation([self.x0], np.array([2, 3]))
        msg = DpopMessage('UTIL', u1_0)
        self.a0._on_util_message(self.x1.name, msg, 0)

        # root only received one message, it should not send any message yet
        self.assertEqual(self.sender0.value_msg_data, None)
        self.assertEqual(self.sender0.util_msg_data, None)

        u2_0 = NAryMatrixRelation([self.x0], np.array([5, 3]))
        msg = DpopMessage('UTIL', u2_0)
        self.a0._on_util_message(self.x2.name, msg, 0)

        # a0 is the root, it has received UTIL message from all its children:
        #  it must compute its own optimal value
        self.assertEqual(self.sender0.value_msg_data, ([self.x0], ['a']))
        self.assertEqual(self.a0.current_value, 'a')

    def test_value_leaf_two_vars(self):
        self.a0.on_start()
        self.a1.on_start()
        self.a2.on_start()

        msg = DpopMessage('VALUE', ([self.x0], ['a']))
        self.a1._on_value_message(self.a0, msg, 0)
        self.assertEqual(self.a1.current_value, 'b')

        msg = DpopMessage('VALUE', ([self.x0], ['a']))
        self.a2._on_value_message(self.a0, msg, 0)
        self.assertEqual(self.a2.current_value, 'a')


class SmartLigtSampleTests(unittest.TestCase):

    def test_4variables(self):
        l1 = Variable('l1', list(range(10)))
        l2 = Variable('l2', list(range(10)))
        l3 = Variable('l3', list(range(10)))
        y1 = Variable('y1', list(range(10)))

        @AsNAryFunctionRelation(l1, l2, l3, y1)
        def scene_rel(l1_, l2_, l3_, y1_):
            if y1_ == round((l1_ + l2_ + l3_) / 3):
                return 0
            return 10000

        @AsNAryFunctionRelation(l3)
        def cost_l3(l3_):
            return l3_

        self.assertEqual(scene_rel(9, 6, 0, 5), 0)

        self.assertEqual(scene_rel(3, 6, 0, 5), 10000)

        joined = dpop.join_utils(scene_rel, cost_l3)

        self.assertEqual(joined(9, 6, 0, 5), 0)

        self.assertEqual(joined(3, 6, 0, 5), 10000)

        util = dpop.projection(joined, l3, 'min')

        # print(util)
