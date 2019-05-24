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

import pydcop.dcop.relations
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


class DummySender(object):
    def __init__(self):
        self.util_sender_var = None
        self.util_dest_var = None
        self.util_msg_data = None
        self.value_sender_var = None
        self.value_dest_var = None
        self.value_msg_data = None

    def __call__(self, sender_var, dest_var, msg, prio=None, on_error=None):
        if msg.type == "UTIL":
            self.util_dest_var = dest_var
            self.util_msg_data = msg.content
        elif msg.type == "VALUE":
            self.value_dest_var = dest_var
            self.value_msg_data = msg.content


@pytest.mark.skip
class TestAlgoExampleTwoVars:
    """
    Test case with a very simplistic setup with only two vars and one relation
     a0 -> a1

    """

    def setup_method(self):
        self.x0 = Variable("x0", ["a", "b"])
        self.x1 = Variable("x1", ["a", "b"])

        self.r0_1 = NAryMatrixRelation([self.x0, self.x1], np.array([[1, 2], [4, 3]]))

        self.sender0 = DummySender()
        self.sender1 = DummySender()
        compdef = MagicMock()
        compdef.algo.algo = "dpop"
        compdef.algo.mode = "max"
        self.a0 = dpop.DpopAlgo(
            self.x0,
            parent=None,
            children=[self.x1.name],
            constraints=[],
            comp_def=compdef,
        )
        self.a1 = dpop.DpopAlgo(
            self.x1,
            parent=self.x0.name,
            children=[],
            constraints=[self.r0_1],
            comp_def=compdef,
        )

        self.a0.message_sender = self.sender0
        self.a1.message_sender = self.sender1

    def test_onstart_two_vars(self):

        # a0 is the root, must not send any message on start
        self.a0.on_start()
        assert self.sender0.util_msg_data is None
        assert self.sender0.value_msg_data is None

        # a1 is the leaf, sends a util message
        self.a1.on_start()
        print(self.sender1.util_msg_data)

        assert self.sender1.util_msg_data("a") == 2
        assert self.sender1.util_msg_data("b") == 4

    def test_on_util_root_two_vars(self):

        # Testing that the root select the correct variable when receiving
        # the util message from its only child.

        self.a0.on_start()
        self.a1.on_start()

        u1_0 = NAryMatrixRelation([self.x0], np.array([2, 4]))
        msg = DpopMessage("UTIL", u1_0)
        self.a0._on_util_message(self.x1.name, msg, 0)

        # a0 id the root, when receiving UTIL message it must compute its own
        #  optimal value and send a value message
        msg = DpopMessage("VALUE", ([self.x0], ["b"]))
        assert self.sender0.value_msg_data == msg.content
        assert self.a0.current_value == "b"

    def test_value_leaf_two_vars(self):

        self.a0.on_start()
        self.a1.on_start()

        msg = DpopMessage("VALUE", ([self.x0], ["b"]))
        self.a1._on_value_message(self.a0, msg, 0)
        assert self.a1.current_value == "a"


@pytest.mark.skip
class TestAlgoExampleThreeVars:
    """
    Test case with a very simplistic setup with only two vars and one relation
     a0 -> a1
        -> a2

    """

    def setup_method(self, method):
        self.x0 = Variable("x0", ["a", "b"])
        self.x1 = Variable("x1", ["a", "b"])
        self.x2 = Variable("x2", ["a", "b"])

        self.r0_1 = NAryMatrixRelation([self.x0, self.x1], np.array([[1, 2], [2, 3]]))
        self.r0_2 = NAryMatrixRelation([self.x0, self.x2], np.array([[5, 2], [3, 1]]))

        self.sender0 = DummySender()
        self.sender1 = DummySender()
        self.sender2 = DummySender()
        compdef = MagicMock()
        compdef.algo.algo = "dpop"
        compdef.algo.mode = "max"

        self.a0 = dpop.DpopAlgo(
            self.x0,
            parent=None,
            children=[self.x1.name, self.x2.name],
            constraints=[],
            comp_def=compdef,
        )
        self.a1 = dpop.DpopAlgo(
            self.x1,
            parent=self.x0.name,
            children=[],
            constraints=[self.r0_1],
            comp_def=compdef,
        )
        self.a2 = dpop.DpopAlgo(
            self.x2,
            parent=self.x0.name,
            children=[],
            constraints=[self.r0_2],
            comp_def=compdef,
        )

        self.a0.message_sender = self.sender0
        self.a1.message_sender = self.sender1
        self.a2.message_sender = self.sender2

    def test_on_start(self):
        # a0 is the root, must not send any message on start
        self.a0.on_start()
        assert self.sender0.util_msg_data is None
        assert self.sender0.value_msg_data is None

        # a1 is a leaf, sends a util message
        self.a1.on_start()
        print(self.sender1.util_msg_data)

        assert self.sender1.util_msg_data("a") == 2
        assert self.sender1.util_msg_data("b") == 3

        self.a2.on_start()
        print(self.sender2.util_msg_data)

        assert self.sender2.util_msg_data("a") == 5
        assert self.sender2.util_msg_data("b") == 3

    def test_on_util_root_two_vars(self):
        # Testing that the root select the correct variable when receiving
        # the util message from its only child.

        self.a0.on_start()
        self.a1.on_start()

        u1_0 = NAryMatrixRelation([self.x0], np.array([2, 3]))
        msg = DpopMessage("UTIL", u1_0)
        self.a0._on_util_message(self.x1.name, msg, 0)

        # root only received one message, it should not send any message yet
        assert self.sender0.value_msg_data == None
        assert self.sender0.util_msg_data == None

        u2_0 = NAryMatrixRelation([self.x0], np.array([5, 3]))
        msg = DpopMessage("UTIL", u2_0)
        self.a0._on_util_message(self.x2.name, msg, 0)

        # a0 is the root, it has received UTIL message from all its children:
        #  it must compute its own optimal value
        assert self.sender0.value_msg_data, ([self.x0], ["a"])
        assert self.a0.current_value == "a"

    def test_value_leaf_two_vars(self):
        self.a0.on_start()
        self.a1.on_start()
        self.a2.on_start()

        msg = DpopMessage("VALUE", ([self.x0], ["a"]))
        self.a1._on_value_message(self.a0, msg, 0)
        assert self.a1.current_value == "b"

        msg = DpopMessage("VALUE", ([self.x0], ["a"]))
        self.a2._on_value_message(self.a0, msg, 0)
        assert self.a2.current_value == "a"


class TestSmartLightSample:
    def test_4variables(self):
        l1 = Variable("l1", list(range(10)))
        l2 = Variable("l2", list(range(10)))
        l3 = Variable("l3", list(range(10)))
        y1 = Variable("y1", list(range(10)))

        @AsNAryFunctionRelation(l1, l2, l3, y1)
        def scene_rel(l1_, l2_, l3_, y1_):
            if y1_ == round((l1_ + l2_ + l3_) / 3):
                return 0
            return 10000

        @AsNAryFunctionRelation(l3)
        def cost_l3(l3_):
            return l3_

        assert scene_rel(9, 6, 0, 5) == 0

        assert scene_rel(3, 6, 0, 5) == 10000

        joined = pydcop.dcop.relations.join(scene_rel, cost_l3)

        assert joined(9, 6, 0, 5) == 0

        assert joined(3, 6, 0, 5) == 10000

        util = pydcop.dcop.relations.projection(joined, l3, "min")

        # print(util)
